# Closed-loop reference input: fixed five-arm test

| progress | The reference-window intervention and actor-input trace are implemented; source review repairs precede execution. |
|---|---|
| bottleneck | Matched-state action sensitivity is not closed-loop behavioral evidence. |
| next step | Run O2, then four-state O4b, as separate fixed five-arm artifacts. |

## Inputs and stop rules

| Order | Exact retained config SHA-256 | Same-config positive-control manifest SHA-256 |
|---|---|---|
| O2, legacy three-state vocabulary | `af616f2fd9c1ec36f1db59812e93e5b5e91e89f013d818d9a3ca293032b6f947` | `457fceb2b8ded1c0e933fec15c540289121a472066852c1ee9530e15fde55034` |
| O4b, four-state vocabulary with unused rise slot | `82df728a0dd073c07e0bdcd0fae3579893986d63277a2e924b2d09f7e0557fae` | `3c6764a695331d94edcb9aae44566cd4d50cb4156e9087f0abce70ee0e9bd461` |

- Positive controls live under sibling `humanoid-harness-probe-runs/`:
  `gmt_course_o2_probe_v1_20260906` and
  `gmt_course_o4b_fourstate_probe_20260907`. Their retained output hashes
  re-verified before this protocol was committed.
- Use their exact retained input-config bytes. Do not combine vocabulary
  families into one artifact or reuse one family's initial residual policy.
- Each exact arm must match its control's trajectory, frame ledger and
  evaluation bytes. Any mismatch invalidates that artifact's causal readout;
  stop before another launch and diagnose. Do not normalize away differences.
- Every arm must have byte-identical initial `qpos` and `qvel`, verified
  fixed-home reset metadata, fresh actor history, literal zero residual and
  unchanged original objective targets. Verify all actual actor observations
  and recompute outputs from the pinned numeric actor weights.
- Source/config/asset/runtime/resource mismatches stop interpretation. Keep
  failures and all completed arms; no substitute seeds or added treatments.
- One local job at a time; maximum 1,200 s, 8 GiB RSS per artifact. No training.

## Locked interventions and predictions

| Arm | Only direct intervention | Prediction within first 50 steps |
|---|---|---|
| exact | no window change | byte parity with the positive control |
| zero_reference | positive float32 zeros in all 600 reference slots | practical divergence |
| current_frame_repeated | repeat the pre-step current frame in all 20 slots | genuinely uncertain |
| shuffled_reference | fixed PCG64(20260906) permutation of the 20 rows | practical divergence |
| shifted_reference | active segment at reported phase + native offsets + 5 s | practical divergence |

- The shuffle permutation is fixed across steps, episodes and both configs;
  it is not freshly randomized on reset. Shift never invents a future mode.
- Current-frame repetition removes lookahead, not all temporal information.
  Its pre-step frame differs from the post-step objective target.
- Practical divergence: root translation >=1 mm, any joint angle >=1 mrad,
  or sign-invariant root-orientation geodesic >=1 mrad, versus exact.
- Report first 50 steps and the full common observed horizon; comparison
  after either arm terminates is unavailable. Report first different action,
  first practical trajectory difference, first different oracle command,
  survival/fall, and original-target versus actor-window-first-row errors.
- The original oracle uses each arm's own evolving state. Its later modes,
  phases, transitions and proprioceptive history may therefore diverge. These
  are consequences of the input intervention, not additional direct edits.

## Interpretation

```text
fixed config + admitted references → original state-aware oracle
                                              ↓ original command
                                  [fixed actor-window treatment]
                                              ↓ actual 2154-value input
                                   frozen actor → fixed G1 dynamics
                                       ↑                 ↓
                                       └── own history/state ──┘
original post-step target + recorded trajectory → independent metrics
```

- Divergence establishes input dependence only for the tested config and
  interventions. It does not prove good tracking, composition quality,
  successful learning, held-out robustness or universal reference use.
- Corrupted-input failure can demonstrate dependence; following the corrupted
  target successfully is not required. A null is only no detected effect under
  the specified intervention, not universal non-use.
- Raw substep contact/saturation records and sensor values remain identified
  producer evidence; data-only validation does not rerun physics.
- This diagnostic artifact cannot enter ordinary task-score or training-result
  registries. Its result informs the next experiment; it is not a task win.
