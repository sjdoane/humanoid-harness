# G1 learning loop — measured results

| status | evidence |
|---|---|
| progress | 13 bounded training runs completed; reference and reward revisions were proposed, admitted, trained and evaluated. |
| bottleneck | No policy passes the full posture-course gate. Depth and heading remain unresolved. |
| next step | Test earlier state-triggered crouch entry with reward r1; retain the failed reward revision as evidence. |

## Architecture actually used

```text
INPUT: task + admitted motion library + verified development traces
                            ↓
            LLM proposal: oracle OR task reward
                            ↓
       data-only validation + immutable candidate identity
                            ↓
robot state → oracle → future reference window → frozen GMT base actor
     ↑          └── phase/mode ──→ learned residual ← task observations
     │                                     ↓
     └──────────── fixed G1 plant ← base action + bounded residual
                                            ↓
         OUTPUT: states, contacts, tracking errors, task metrics, video
                                            ↓
          verified feedback → LLM diagnosis → next single-factor proposal
```

- Frozen: G1 dynamics/reset, observations/actions, base weights, tracking reward,
  PPO configuration, per-run budget, evaluator and final-checkpoint selection.
- Authored: reference segments/state guards/entry phase; task-reward recipe.
- LLM calls occur between runs. Deterministic control runs at 50 Hz; PD at 1 kHz.
- Oracle transitions use measured progress. Phase matching currently occurs at
  behavior entry; within a segment, playback still advances on a local clock.
  This is **not** continuous learned phase estimation.
- Frozen base weights do not freeze closed-loop behavior: executed composite
  actions feed the base actor's history. The residual's immediate target bound
  is ±0.125 rad, not a bound on cumulative behavior.

## Why this MDP

- GMT/G1 supplies a compatible reference-conditioned controller and motion data.
- Two real supplied-motion baselines established local tracking before learning.
- Gymnasium remains the learning interface; native Humanoid-v5 stays a separate
  experiment family. No cross-robot comparison is treated as a treatment effect.
- The blue posture region is a task constraint on flat ground, **not** an obstacle.
  No locomanipulation, collision-avoidance or original-JIT-equivalence claim.
- Research basis: [GMT](https://github.com/zixuan417/humanoid-general-motion-tracking),
  [OGMP](https://arxiv.org/html/2403.04205v3),
  [Eureka](https://arxiv.org/html/2310.12931v2).
- RL-Sculptor's useful boundary is retained: author the oracle/task reward, while
  keeping tracking, MDP and independent objective evaluation outside author control.

## What the controls show

- Execution source: `004592f5af1b9e1580cb0925890876efe4f48496`.
- Seeds: `20260906`, `20260907`, `20260908`; 32,768 transitions per arm/seed.
- Same initial policy within each seed; final checkpoint only; one fixed start.
- O0: walk in all slots. O2: walk → crouch → walk with entry-window/terminal-hold
  semantics. r0: tracking only. r1: tracking plus the initial task recipe.

| trained arm | full 20 s, no fall | inside posture compliance | inside mean-speed error | lateral max, full-horizon runs | full task passes |
|---|---:|---:|---:|---:|---:|
| O0r0 | 3/3 | 0% | 0.007–0.043 m/s | 1.28–2.17 m | 0/3 |
| O0r1 | 3/3 | 0% | 0.030–0.070 m/s | 2.35–7.36 m | 0/3 |
| O2r0 | 1/3 | 46.4–50.7% | 0.062–0.243 m/s | 3.76 m; other two fell | 0/3 |
| O2r1 | 3/3 | 49.3–58.5% | 0.008–0.110 m/s | 1.66–4.85 m | 0/3 |

- In this three-seed development comparison, r1 reduces inside mean-speed error
  versus r0 on O2 in all three seeds; survival is better in two and tied in one.
- On O0, r1 worsens lateral drift in all three seeds. It is not a generally
  better reward. Walking alone never supplies the required crouch.
- Do not compare failed-run lateral maxima or whole-run means as equal exposure:
  the two O2r0 failures lasted only 6.86 s and 4.76 s.
- Three policy seeds do not establish robustness to new starts, motions or tasks.

## Actual revision history

| revision | measured outcome | decision |
|---|---|---|
| O1: unrestricted crop entry/wrap | falls after 4.12 s | retain failure |
| O2: entry window + terminal hold | 20 s, both switches, no fall; still misses task gates | keep development oracle |
| O2b: state-ready exit guard | cannot satisfy exit; falls after 5.46 s | reject |
| r2: LLM raises speed weight 1.0 → 1.5 | mean speed improves, but consistency/posture/drift worsen | reject; retain r1 |

- O1 replay under runtime v2 is byte-identical to its v1 trajectory and frames.
  The O1/O2 comparison is not explained by an unintended default-runtime change.
- O2r2, fresh seed-20260906 training: inside mean speed `0.648 m/s`, but inside
  speed MAE `0.258` versus `0.149`; overall MAE `0.455` versus `0.307`;
  posture compliance `48.7%` versus `58.5%`; lateral drift `8.61` versus `2.97 m`.
- The reward revision closes a real negative-result loop: verified feedback →
  retained LLM proposal → independent admission → fresh training → rejection.
  It does not demonstrate task success.

## Diagnosis and guardrails

- O2r1, seed 20260906: reference posture is compliant in `46/65` actual-region
  samples; robot posture in `38/65`. Timing accounts for much of the mismatch.
- Actual-minus-reference height bias averages `+0.044 m`; at the minimum,
  actual height is `0.509 m` while reference height is `0.442 m`.
- The current task posture reward is flat below `0.60 m`. It cannot specifically
  encourage the evaluator's `≤0.50 m` dip. Do not move that evaluator threshold.
- Reject a reference-relative height term disguised as task reward: that would
  retune tracking. Any future task-height term must use the independent task.
- Progress-weighting is not a general anti-gaming proof under discounting,
  stopping or backtracking. Test the actual failure modes.

## Reproducible evidence and tools

- Local artifact root: sibling `humanoid-harness-probe-runs/`.
- Three-seed index: `artifacts/gmt/course_configs/gmt_course_three_seed_results_v1.json`.
  SHA-256: `b7104c4885511e597e3410dd7bd0449f7443889465bc28a2643c2a817734715f`.
- Every output digest rechecked; all 12 objective reports recomputed; initial
  policies match within seed; base digests unchanged; zero-residual traces match
  their probes byte-for-byte. All workers cleaned up and released their slots.
- O2r1 replay: `gmt_course_o2r1_task_overlay_v1_recorded.gif`.
  SHA-256: `8ac1aff9e23f808839cb3b9406065165c1bf4d5c88f89b5cb85d3267e9793db2`.
- r2 source manifest: `3cdb12be3194ab7737a68a648610aa25de0c13b4e952422e0e97578a093e921f`.
- Exact LLM reply is retained in `gmt_course_r2_author_mailbox.json`; its body
  SHA-256 is `ca353bbfe6611cbcbe364a2d62a80266ba4a7250cf50a2fd3c0ad4c7dfef93a6`.
  Client-declared authorship is retained; served-model attestation is not claimed.
- The new feedback command verifies the pinned output ledger, trajectory
  crosslinks, boundary-derived metrics and objective recomputation. Transient
  substep failures remain producer-recorded evidence, not independently recreated.

```bash
humanoid-harness g1 feedback \
  --manifest /absolute/path/course_run_manifest.json \
  --manifest-sha256 EXPECTED_SHA256 \
  --label final_policy --output /absolute/new/feedback-directory
```

- Reporting/visualization changes after the study do not alter its training
  runtime: `b609bf6`, `6d30af1`, `a8f7431`. Parent checks: 200 focused tests,
  9 CLI tests, Ruff and compile checks pass.
- Astra leads implementation and experiments. Fable supplies review and ideas.
  Original motion/controller downloads were not executed as uploaded code.
