# After-only state-feedback oracle

| progress | O7b isolates a surviving four-stage trajectory with no later region revisit. |
|---|---|
| bottleneck | It still veers 10.94 m sideways. Neither the full task nor the after-crop screen passed. |
| next step | One fixed-law, zero-residual steering probe after a fresh legacy-control reproduction. |

## Question and boundary

- Can a bounded oracle correction, computed from observed heading and lateral
  displacement, control the after-phase drift of this exact trajectory?
- **New runtime/oracle capability family**, not adoption of O7b, a reward study,
  training, or a general-purpose learned phase estimator.
- Two arms at one final, reviewed source: unchanged legacy O7b; opt-in feedback
  profile with exactly the same task, motion crops, guards, dwell and seed.
- No source or simulator change during the pair. Pin the final profile ID,
  exact configs, executable tree and native reservations before execution.

```text
INPUT: pre-action robot boundary + fixed task frame + native H×30 window
                       ↓
  after mode only: lateral → target heading → yaw correction + total clamp
                       ↓
           OUTPUT: corrected H×30 native window
                       ↓
              frozen GMT actor → fixed G1 plant
                       ↑                  │
                       └── measured state┘
retained states/commands → independent scorer → next LLM proposal (between runs)
```

- Frozen: actor weights/history ABI, MDP/reset/termination, residual=0,
  observations/actions, original tracking/task rewards, evaluator, 50 Hz command
  and native motion cadence, four oracle states and 1,000-step horizon.
- Feedback uses the observed **pre-action** boundary, not target orientation or
  post-action state. The native motion remains immutable; transformed windows
  receive their own trace identities.
- Both coordinates use the evaluator's reset-anchored `TaskFrame`; lateral and
  heading are left-positive. Required sign fixture: `y=+1 m, psi=0` gives
  target `-0.3 rad` and a negative correction, including in a rotated task frame.

## Fixed law (no gain search)

For task-frame lateral displacement `y` in metres and heading `psi` in radians:

```text
target_heading = clip(-0.3 rad/m * y, -0.3 rad, +0.3 rad)
correction     = 1.0 /s * wrap(target_heading - psi)
issued_yaw[j]  = clip(native_yaw[j] + correction, -0.3, +0.3) rad/s
```

- Apply to column 6 of every future native row; preserve the other 29 columns,
  float32 shape/order, cadence, and source window. Active only in `after`.
- Arithmetic: compute target/wrapped correction from float64 state; retain that
  analytic correction and its float32 applied value. Add the applied correction
  to native float32 yaw, clip with float32 limits, and retain float32 output.
  The held poststep target uses the identical operation order.
- **Correction plus total-rate clipping is the combined manipulation.** The
  clamp can also clip native content above 0.3 rad/s. Report both native-limit
  exceedance and output saturation; do not claim correction-only causality.
- Hold that exact pre-action correction for the matching `current_after_step`
  reference. Do not recompute it from the resulting state or next prepared mode.
- Preserve both existing native clock paths. At a float32 crop-wrap boundary,
  `window[0]` can differ from separately recomputed `current_after_step`.
  Apply the same held correction independently to both authoritative targets;
  reconstruct and report their equality or mismatch, **do not force equality**.
  A wrap-boundary regression is required. No endpoint-clock repair is hidden
  inside this yaw-feedback intervention.
- At the retained entry boundary, `psi=+0.092443 rad` and `y` is about `+1.42 m`:
  initial correction is about `-0.392443 rad/s`. Native rows below `+0.092443`
  therefore saturate at `-0.3 rad/s`. An immediate handover fall is a meaningful
  rejection; neither survival nor smooth recentering is presumed.
- Record pre-action heading/lateral, target/correction, before/after window
  hashes and saturation. Reconstruct these from raw states and pinned native
  motion in validation; self-reported receipts alone are insufficient.
- Disabled profiles emit no added trace fields and preserve historical bytes.
  No automatic training admission, exposure telemetry or phase-rate changes.

## Controls and rejection rules

- Both fresh arms use the reviewed 195-file executable tree
  `c4f5a24e285c01bde06d0c5530631084284eacdf22106d614b6e5ad228d0174d`.
  Native resource acceptances bind the exact final clean commit separately.
- Before data, a 25-case independent arithmetic grid checks all 600 window
  values, unchanged channels, held endpoint, counts, fractions and array hashes.
- Retained O7b config SHA:
  `ba45dba36ef88bdc522ed2115e46a4b3d876e00a627a089fd56ddf0de623e18a`.
- Retained manifest SHA:
  `92e2a72f743bded8b2694760c7492f8d35f0829489415a05c3afff6ec459944a`.
- Candidate config SHA:
  `8fe33d993c6226690e986e99de3d0d1ee436424239069c4cceb91c1a8fabfc8e`.
  Its only changes are schema version 4 and profile
  `gmt_g1_four_state_loop_after_heading_feedback_course/v1`.
- Saturation means the unclipped float32 total is **at or above** either
  absolute limit; native-limit exceedance is strictly above the limit.
- First run the legacy control at new source. All three evidence outputs must
  reproduce byte-for-byte before enabling feedback. No metadata exception here.
- Enabled arm: seed `20260906`, zero residual, 1,000 steps, one bounded job.
  Require unchanged first 263 frame records/actions and 264 qpos/qvel rows.
  Only the declared reset profile metadata may differ before the intervention;
  no reward, numerical state or command exception.
- Rebuild objective/feedback independently and verify native reservation,
  exact runtime source, inputs and outputs. Integrity failure invalidates the
  result; it is not a negative behavioral trial.

| Predata screen | Criterion |
|---|---|
| P1: lateral containment | Whole-run maximum absolute lateral error <2.5 m |
| P2: after heading | Maximum absolute unwrapped heading during after <0.5 rad relative to reset |
| Survival | 20 s, no fall |
| Composition | Three switches, rise and after executed |
| Traversal | Finish reached at 3.5 m; no region `[1,2)` re-entry after after-mode entry |
| Tracking | Joint p95 <=0.35 rad; roll-pitch p95 <=0.25 rad |

- Unwrap the full boundary heading series before selecting after rows. Entry
  is qpos row 263, not first post-action row 264. Report entry-relative excursion
  separately and the first two seconds after entry separately; neither replaces P2.
- Report all eleven original task gates unchanged, the full lateral/heading
  path, issued yaw integral/series, saturation, after speed MAE, first-visit
  and later-region exposure. All physical region visits remain authoritative.
- **No full-task success predicted:** lateral error already exceeds the original
  0.75 m limit before after entry; unchanged crouch minimum is 0.515773 m,
  above the original 0.50 m threshold. Later recovery cannot erase earlier error.
- Any failed screen rejects this fixed law for advancement. Success qualifies
  this after-only primitive, not other modes, datasets, starts or task success.
  Failure does not disprove other gains/crops/feedback laws. No automatic sweep.
- Fable reviewed the fixed law and corrected crop/entry-boundary interpretation
  in `20260907T064917.062827Z-f1e40f52b9af4c0fa791d615b41439f3`.
  Total-clamp decision: `20260907T065600.360602Z-816398d359c3430591c12bf36d07e7e8`.
