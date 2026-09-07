# G1 learning loop — measured results

| progress | 34 bounded training runs / 1,900,544 transitions. The fixed-normalizer O2 policy survives 20 s with both switches. |
|---|---|
| bottleneck | No full task pass. Study 011 misses its compliance floor and three task gates; no trainer default is changed. |
| next step | Diagnose posture timing/depth and lateral drift; choose the next bounded oracle/reward comparison from verified feedback. |

- [Study 011: fixed input normalization](../../../experiments/011_g1_fixed_normalization/RESULTS.md):
  exact control reproduction; candidate survives, but 64.2% compliance misses
  the locked 68.35% floor. Retained, not adopted. Normalization alone is not a
  complete-task result; conditioning and transition feasibility remain distinct.

- [Study 009: actual reference use](../../../experiments/009_g1_reference_input/RESULTS.md):
  two five-arm interventions change actions and trajectories; exact controls
  reproduce retained bytes. No learning or composition-quality claim.
- [Study 010: after-only crop](../../../experiments/010_g1_after_reference/RESULTS.md):
  unchanged-prefix parity passes, robot falls at 6.76 s; candidate rejected.
- [Study 008: repeatable crouch](../../../experiments/008_g1_repeatable_crouch/RESULTS.md):
  falls at 6.4 s before rising; probe-only profile remains ineligible for training.
- [Study 007: lower learning rate](../../../experiments/007_g1_learning_rate/RESULTS.md):
  both predefined adoption criteria fail. No successful task policy is established.
- These are development results. The earlier studies below retain their
  original settings and evidence; they are not current launch instructions.

- Latest: [composition x depth results](../../../experiments/006_g1_composition_depth/RESULTS.md).
  Three of four larger arms fall; one stops. All failures are retained.
- Next: [fixed learning-rate test](../../../experiments/007_g1_learning_rate/PROTOCOL.md).
  A repeatable crouch/rise compositor is a separate, initially probe-only family.
- The following trainer-screen and original factorial records remain valid
  historical results, not the current best-task claim.

- Latest: [trainer screen](../../../experiments/005_g1_training_conditioning/RESULTS.md)
  and [fixed replication jobs](../../../experiments/005_g1_training_conditioning/REPLICATION.md).
- Scaling passes the screen on all three r1 seeds. Lateral error is 0.526 m
  on seed 20260906; full task still fails. The matched scaled r0/r1 contrast
  favors r1 for overall speed MAE and lateral drift, not inside speed in every seed.
- O4 reaches 83.1% posture compliance without a residual but falls at 5.4 s.
  It is rejected for learning admission, not promoted from its posture score.
- O4b adds an observed-height exit condition: 20 s survival, both switches,
  unchanged 83.1% compliance. It passes gait admission but still misses three
  full-task gates. Phase matching remains entry-only, not continuous estimation.
- The G1 UI validates 12 explicitly registered snapshots through the same
  feedback/evaluator path. It is a selected evidence view, not the complete run ledger.

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
| O3: LLM advances entry guard 0.65 → 0.30 m | trained posture compliance rises to 72.7%, but falls at 6.96 s | reject; retain O2 |
| r3: LLM raises heading weight 0.5 → 3.0 | lateral drift grows from 2.97 to 8.34 m; heading predictions and speed-MAE guard fail | reject; retain r1 |

- O1 replay under runtime v2 is byte-identical to its v1 trajectory and frames.
  The O1/O2 comparison is not explained by an unintended default-runtime change.
- O2r2, fresh seed-20260906 training: inside mean speed `0.648 m/s`, but inside
  speed MAE `0.258` versus `0.149`; overall MAE `0.455` versus `0.307`;
  posture compliance `48.7%` versus `58.5%`; lateral drift `8.61` versus `2.97 m`.
- The reward revision closes a real negative-result loop: verified feedback →
  retained LLM proposal → independent admission → fresh training → rejection.
  It does not demonstrate task success.

## Diagnosis and guardrails

### Reward-family incentive audit (2026-09-07)

Independent read-only audit and parent recomputation at `57d2a2e`:

| Analytic task-reward setting, before tracking | Reward / maximum |
|---|---:|
| R1: stationary, aligned, outside the posture region | 3.50432 / 4.5 (77.87%) |
| R4: inside, height 0.60 m, speed 0.65 m/s, aligned | 3.31967 / 4.5 |
| R4: immediately outside, same height/speed/alignment | 4.47260 / 4.5 |
| R4 with speed weight 5: stationary/aligned/outside | 3.52160 / 8.5 (41.43%) |

- These are counterfactual feature evaluations of the actual reward function,
  not rollouts or proofs that those policies are dynamically achievable.
- Outside-region posture error is zero, so the posture term pays +2 per living
  step in R1/R4. There is no explicit progress, finish or milestone term.
- Separate observed counterexample: O4b/R1 at 131,072 transitions survives but
  stalls near final progress 1.11370 m. Its 733 inside-region frames average
  0.008356 m/s. This is an **inside** stall, not the outside analytic example.
- Source: `adapters/gmt/course_task.py`; retained run:
  `gmt_course_o4br1_scale64_131072_seed20260906_20260907`.
- The family can alter local incentives but does not explicitly express task
  traversal. This is a concrete design concern, not an isolated causal diagnosis.
- Finish the fixed-normalization comparison before a separate reward test.
  A higher speed weight is a candidate expressiveness test, not an adopted
  reward; prior R2 improved mean speed while worsening other task measures.

### Earlier revision safeguards

- O3 execution source: `1b501d6`; 32,768 transitions, seed `20260906`.
  Control/training code is unchanged from `004592f`; only reporting tools differ.
- O3's five preregistered trained predictions fail: entry phase `1.240 < 1.25 s`,
  reference compliance `49/55 = 0.891 < 0.90`, robot compliance `0.727 < 0.75`,
  fall/exit mismatch, and speed/roll-pitch failures. No near-miss is a pass.
- O3's zero-residual 80.4% posture compliance is a separate descriptive result;
  that rollout also falls, at 5.36 s. It cannot substitute for trained predictions.
- O3 manifest: `dc692f286485523031be94a23f06ff5b2b6dccc66a5aba0515d3e8cb6ea29f17`.
  Verified feedback: `1a9a120bcb9ec2b3b3e6897993f928ec0bed1f748b8a74ec674840691cdcdc14`.
- The original 32,768-step pilots did not retain optimizer curves. A later
  measurement-only reproduction records them without changing policy behavior.

## Trainer diagnosis, not a successful policy

- Source `9188ed0`: O2r1 compatibility control reproduces all nine original
  artifacts byte-for-byte. Optional reward v2 is disabled; reporting/revision
  changes do not explain r3's outcome.
- r3: 32,768 transitions, seed `20260906`, no fall over 20 s, both switches.
  Heading absolute mean `0.942` versus `0.506 rad`; lateral max `8.342` versus
  `2.970 m`; speed MAE `0.424` versus `0.307 m/s`; compliance `56.2%` versus `58.5%`.
  Its actual candidate came from `g1 revise`, with exact LLM input/output lineage.
- Source `0923a5a`: telemetry control again reproduces all nine original outputs
  byte-for-byte, including both policies and complete recorded trajectories.
  It adds 65 telemetry records; 45 training episodes and 18 falls are unchanged.
- Across 64 updates: explained variance ranges `-0.0110…0.0219` (last `0.00000745`);
  value loss starts `4895.65` and ends `4884.77`; mean KL `0.0281`; mean clipping
  fraction `0.186`. Attempted epochs: 149 of 256 nominal. These include epochs
  stopped partway through; completed epochs/minibatches were not counted.
- This supports testing training conditioning. It does not identify critic
  dominance: actor/critic networks are separate, and pre-clip gradient norms
  were not measured. Global gradient clipping can couple the two branches.
- Opt-in scaling is a **trainer experiment**, not an oracle/reward improvement.
  Keep task semantics, relative reward weights and evaluator thresholds fixed;
  compare matched budgets and do not compare value-loss magnitudes across units.
- Practical basis: [SB3 RL guidance](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)
  recommends checking preprocessing and sample budget for custom environments.

| retained artifact | SHA-256 |
|---|---|
| r3 run manifest | `71c6b970455f28444d0459c1d387682aa09476a55a6d184b271437f20d01af23` |
| r3 revision receipt | `a46ea453febfcc9e7de34f6f093f3b79d05c48d875e4872a1845ec7f8081bb3d` |
| telemetry-control manifest | `3b4d391af582fb1772a45055d79530b8719ab843e722f1f80f0a10b15273001c` |
| training telemetry | `a4c12a940d14f00558b7e9f74ab0c2306b70f477c21d0768a8c26ccf1911dcb3` |
| telemetry-aware feedback | `c2e72807443aeec67f8e2b024b71b5f7f965fa66d3713a8a50a91fb9dbda3b4c` |

- Parent verification at `0923a5a`: **266 focused tests pass**, including actual
  PPO on a numeric fixture. Ruff passes. Whole-repository failures remain separate.

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
