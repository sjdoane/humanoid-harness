# Training degradation: rival explanations and bounded tests

| progress | Study018 trained A is worse than zero residual on task metrics and raw episode reward. |
|---|---|
| bottleneck | One seed cannot isolate exploration, actor drift, reward sensitivity or handover quality. |
| next step | Complete oracle-only019 first; choose a separate training contrast from the retained evidence. |

Targeted primary-source reading, 2026-09-07; not a systematic review, new
experiment authorization, or proof that a cited method transfers to GMT.

| Source and locator | Reported mechanism | Project implication / limit |
|---|---|---|
| [Residual Policy Learning v1](https://arxiv.org/html/1812.06298v1), IV-A/B | Zero last-layer initialization preserves the initial mean; critic-only burn-in is proposed to avoid early degradation from an uninformed critic. | Our mean starts at zero, but sampled actions do not. Burn-in is a possible trainer experiment, not a reward/oracle improvement. Evidence is from actor-critic manipulation; direct PPO transfer is untested. |
| [DeepMimic](https://xbpeng.github.io/projects/DeepMimic/DeepMimic_2018.pdf), 6.1 and11 | Reference-state initialization exposes later motion states; linear phase synchronization is acknowledged as a timing limitation. | Phase-conditioned resets could improve later-stage coverage, but would change the training reset distribution. They cannot silently enter019 or a reward-only pair. |
| [ZPRL v1](https://arxiv.org/html/2605.19919v1), IV-A/B | A learned bottleneck supports residual changes in latent space while the action generator stays frozen. | Structured exploration is a relevant rival to joint-action noise. GMT does not expose this trained bottleneck; adoption would change the controller interface and require offline training. Defer. |
| [Foresight Residual RL v1](https://arxiv.org/html/2607.16506v1), IV-C,V-B,VII-B,VIII | Downstream rollout success trains a terminal-state predictor; backward training uses it to weight subtask rewards. | Measure what state each stage hands to the next, not only whether it ended. Evidence is one simulated assembly task, with explicit decomposition. A predictor requires independent downstream labels; it is not an LLM's confidence score. |

## What the local data support

- [Study018 result](../../../experiments/018_g1_four_state_reward_loop/RESULTS.md)
  establishes a failed deterministic baseline, not the causal diagnosis.
- The critic can fit a worsening on-policy distribution. Its explained
  variance is not a task score or an actor-admission criterion.
- Earlier entry changes phase **and** handover state/history.019 reports that
  full intervention; it does not isolate a pure phase effect.
- Cauchy versus Gaussian course-error kernels are a proposed project test,
  not a method validated by these papers. Keep scales/weights fixed and report
  experienced reward differences. A positive result would not uniquely identify
  reward saturation; a one-seed null would not exclude it.

## Separate candidates, not a combined patch

Post-hoc root readback, 13:08 UTC: all 1,000 lateral samples per trace were
reconstructed from retained qpos and reset `TaskFrame`, matching recorded
metrics within 1e-12 m. Switch handovers below are **pre-action** boundaries;
first-failure times are post-step. Three deterministic traces, no uncertainty
estimate. Evidence: linked Study018 result and
[Study016 receipts](../../../experiments/016_g1_after_feedback/RESULTS.md).

| Trace | First lateral-limit failure | After-entry lateral / heading |
|---|---|---|
| O7b zero residual | Inside, action index213;4.28 s | Index263;1.410961 m /0.092443 rad |
| Study018 trained A | Inside, action index195;3.92 s | Index268;2.305469 m /0.169542 rad |
| Study016 after-only feedback | Inside, action index213;4.28 s | Index263;1.410961 m /0.092443 rad |

- An after-only correction cannot erase an earlier failure on these frozen
  traces. This does **not** prove that retraining with a new reward cannot
  change earlier behavior.
- Earlier steering and mirrored motion are candidate oracle/supply directions,
  not proven necessities. A yaw-free reference does not establish that the
  handover turn is unsteerable. Complete019 before choosing another oracle test.
- Evaluate each retained checkpoint with paired noise before a trainer diagnosis.
  Early/late training episode shares mix changing policies; they are not exact
  initial/final checkpoint evaluations. RPL's burn-in argument concerns DDPG,
  not a demonstrated PPO defect here.

| Candidate | Frozen comparison needed | Discriminating outcome |
|---|---|---|
| Earlier entry guard |019 zero-residual oracle pair |Entry-phase manipulation and unchanged physical task metrics |
| Reward-tail change |Fresh r1 replication versus one recipe-only revision |Heading/lateral response plus survival, posture, speed and tracking guardrails |
| Stochastic policy diagnosis |Retained initial/final policies, paired predeclared noise seeds; no retraining |Whether degradation also appears under each policy's sampled action distribution |
| Trainer repair |Separately declared trainer family and unchanged oracle/reward |Reproducible actor improvement at equal transition budget |
| Downstream handover predictor |Separate training labels and held-out transition outcomes |Calibration and actual downstream success, not predictor score alone |

No new scene, controller, reset distribution, exploration schedule or training
budget is adopted by this note. Preserve the two-knob research boundary.
