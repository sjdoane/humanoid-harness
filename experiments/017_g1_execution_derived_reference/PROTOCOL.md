# Re-track an execution-derived crouch reference

| progress | The exact 106-pose reference and conversion receipt exist; independent review accepted their provenance. |
|---|---|
| bottleneck | Static geometry is not evidence that the frozen tracker can follow the transformed reference. |
| next step | One fresh legacy control and one reference-replacement probe, after exact data admission and source sealing. |

## Question

- Does a supplied reference built from an executed crouch produce a surviving,
  reasonably faithful walk → crouch → rise → walk composition?
- This tests the **whole reference-replacement bundle**, not one isolated
  quaternion, lateral, cadence or phase-matching effect.
- No IK/depth ladder, feedback-law sweep, training or controller modification.

```text
INPUT: pinned Study012 trajectory → declared numeric transform → new reference
                                      ↓
robot state → fixed four-state oracle with replaced inside segment
     ↑                                ↓
     └── fixed G1 plant ← frozen GMT tracker ← H×30 reference window
                                      ↓
OUTPUT: raw states, contacts, commands → independent evaluator → LLM diagnosis
```

## Frozen comparison

- Control: exact legacy O7b config from Study015, SHA
  `ba45dba36ef88bdc522ed2115e46a4b3d876e00a627a089fd56ddf0de623e18a`.
- Same four-state loop runtime, actor/history, 2172 observations, actions,
  residual zero, G1 model/reset/termination, task/reward, seed `20260906`,
  50 Hz control, 1000 actions, evaluator and original eleven task gates.
- Keep all four states/guards/dwells and before/rise/after segments unchanged.
  Study016's rejected heading-feedback profile is **off in both arms**.
- Candidate changes only the inside reference bundle and its declared asset
  provenance: the new archive plus the segment definition below.
- The native crouch archive remains consumed by the unchanged rise segment.
- New clean source and exact native reservations are required. Pin them, plus
  candidate config SHA and scorer SHA, before either probe. Not yet sealed.
- Run control first. It must byte-reproduce all three retained Study015
  candidate outputs; then require the first 92 complete frame/action records
  and 93 qpos/qvel boundaries to remain exact in the candidate.

## New reference semantics

- Archive: `artifacts/gmt/derived_references/study012_inside_passage_v1.npz`.
  SHA `885e4c1b324a9b41a4d176ec5fee9e4bc634226d5ade46634111cb1c3204c259`.
- Sibling conversion manifest SHA
  `1c7edb409579d757ef6657378ab605e6aa013cf60b552630beb914728bec602c`.
- Donor: Study012 `qpos[92:198]`, 106 boundary poses / 105 intervals / 2.10 s.
  Donor control used a learned residual. That residual is **not** loaded here.
- Float32 joints/root height; root x is reset-task-frame progress rebased at
  selection start; root y zero; unit xyzw quaternion reconstructed from donor
  roll/pitch at yaw zero. These are material transforms, not exact replay.
- Cadence 50 Hz. Native `ReferenceMotion` derives velocities; its 19-frame
  smoothing kernel is unchanged, so its time span differs from 30 Hz data.
  Neither smoothness nor zero body-local lateral/yaw velocity is assumed.
- Inside segment: start `0.0`, end `2.0999999046325684`, entry window `0.0`,
  boundary `hold_last_pose_zero_velocity`; no inside loop or loop-exit gate.
- Entry therefore selects the first pose, not a searched deep pose. The spatial
  exit guard still selects rise, but no longer waits for an inside loop wrap.
- Terminal hold is explicit because the reference ends at a mid-stride
  crouched pose (height `0.595443 m`). Report hold exposure and the exact exit
  phase; do not call the hold a balanced or stationary reference.
- Static screen: all 106 output poses have no non-foot ground contacts;
  maximum foot overlap `0.015431 m`. This is not a dynamics certificate.
- Minimum target height `0.504942 m` is **not** a solution to the `0.50 m` task gate.

## Predata feasibility screen

| Condition | Threshold |
|---|---|
| Survival | 20 s, no fall or recorded non-foot ground contact |
| Composition | Three switches; all four modes executed; region entered/exited |
| Progress | Finish observed at 3.5 m; no physical region re-entry in after |
| Exposure | At least 25 inside-region samples |
| Depth tracking | Inside minimum height within 0.020 m of 0.504942203119977 m |
| Tracking | Joint p95 ≤0.35 rad; roll-pitch p95 ≤0.25 rad |

- The 2 cm depth window is a predeclared engineering feasibility tolerance,
  not a confidence interval or a replacement for the original task gate.
- Any failed screen rejects this exact extraction/conditioning/handover for
  automatic training admission. Integrity failure invalidates the trial.
- Report all original gates unchanged, full trajectories, first/late physical
  region visits, selected entry phase, hold duration, all transitions and the
  command/reference reconstruction. Lateral drift is still a known unresolved
  task failure; this is not a full-task qualification screen.
- No successful seed selection. Failure is local to this reference bundle,
  not proof that self-derived references or the tracker are generally unusable.
- A pass permits consideration of a separately reviewed bounded training pair;
  it does not admit an arbitrary external corpus or certify dynamics broadly.
