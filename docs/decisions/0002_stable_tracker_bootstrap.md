# ADR 0002: stable tracker bootstrap

| | |
|---|---|
| progress | A staged tracker-family decision and six pre-admission gates are defined. |
| bottleneck | No admitted tracker is both stable and behaviorally dependent on its numeric reference window. |
| next step | Run E0, then test the residual and expanded-TQC contenders on permanently excluded development seeds. |

**Status:** development decision, not behavioral evidence

**Date:** 2026-09-03

## Decision

- First contender: frozen deterministic TQC locomotion base plus a
  reference-conditioned PPO residual whose deterministic mean is initialized
  to exact zero.
- Fallback: expand the TQC actor with zero-valued reference-input columns and
  continue as reference-conditioned TQC.
- Development control: reference-conditioned PPO from scratch with reference
  state initialization (RSI).
- Rejected claim: a stock TQC actor can be transferred into stock PPO while
  preserving its complete action distribution.
- Selection happens before tracker admission. It cannot support an oracle
  comparison.

```text
same-runtime Tier-D branch references R[t:t+H] ------+
                                                      v
state x[t] ---> frozen deterministic TQC ---> a_base  reference residual PPO
     ^                                                |       |
     |                                                |     delta_a
     |                                                +---+---+
     |                                                    v
     +--------- Humanoid-v5 <--- bounded sum/clip <--- a_exec

After admission, the base, residual, composition, reward, and action ABI are
one frozen compound tracker. Only the oracle may change in a Family-A study.
```

## Why this order

| path | useful property | blocking risk | decision |
|---|---|---|---|
| Frozen TQC + residual PPO | Keeps deterministic locomotion intact at zero residual; isolates the reference-consuming path | Stock TQC is not reference-conditioned, so the residual carries all tracking work; clipping may remove authority | Preferred first contender |
| Reference-conditioned TQC | Can preserve the TQC actor's mean and state-dependent log standard deviation at initialization | Fine-tuning can erase locomotion; stock-reward replay and critics are invalid under the tracking reward | Fallback; retain actor only, start new replay and critics |
| Direct PPO + RSI | Smallest single-policy architecture; matches the existing local PPO action contract | Published humanoid successes mostly use PD targets and large parallel training, not this raw-torque adapter | Development control |
| TQC to stock PPO | None beyond code reuse | TQC has state-dependent mean and log-standard-deviation heads; stock PPO does not share that parameterization | Reject equivalence claim |
| AMP | Can provide a learned motion-style prior | AMP intentionally removes phase synchronization and exact clip tracking | Defer; not a tracker-admission path |
| DAgger/distillation | Covers states induced by the student | Requires a reference-aware expert; stock TQC cannot label reference-dependent actions | Defer until a teacher tracker exists |

One TQC-generated trajectory is insufficient. A tracker can follow its own
state-feedback rollout while ignoring the reference. Tracker development must
therefore include at least two healthy, same-runtime continuations from a
common restored state whose desired futures require different actions.

## Pre-admission gates

| gate | test | claim ceiling / stop rule |
|---|---|---|
| E0: resource | Complete the reviewed 100,000-step, excluded-seed TQC calibration; retain only its resource receipt | Passing shows affordability and finite execution only; failure stops the 20-million-step proposal |
| E1: initialization identity | On one hash-pinned observation set, compare base TQC with zero-residual and expanded-TQC initialization. Record mean, log standard deviation, deterministic action, seeded sampled action, physical control, and tolerances | Any unexplained mismatch rejects a preservation claim; deterministic zero residual does not imply zero stochastic exploration |
| E2: Tier-D replay | Restore predetermined full MuJoCo integration states, apply the recorded action, and reproduce the next state, observation, contacts, cadence, and wrapper counters | `qpos/qvel` alone are not presumed sufficient; failure rejects the motion as a replay-certified reference |
| E3: branch identifiability | Admit at least two healthy forked continuations from the same restored state with distinct future windows and controls | Failure means the corpus cannot distinguish a reference-aware policy from a state-only policy |
| E4: tracker-family screen | Run residual PPO, expanded TQC, and direct PPO+RSI with the same reference bytes, raw-torque ABI, tracking reward, environment interactions, protected evaluator, and excluded development resets | This may eliminate broken families; differing trainers make it development selection, not a single-factor result |
| E5: causal-use gate | With one frozen candidate, compare exact, positive-zero, seeded shuffle, and nonzero time shift at identical restored states, then run the predeclared short-horizon expected-direction check | Failure blocks tracker admission and every oracle-performance claim |

Use a predeclared lexicographic rule: pass every correctness and safety gate,
then choose the less complex passing contender. Do not select a weighted winner
after inspecting outcomes. Development seeds and resets remain excluded from
the formal oracle comparison.

## Frozen boundary after selection

The admitted tracker identity binds:

- base and residual checkpoint bytes, or the complete expanded-TQC checkpoint;
- architecture, normalizers, actor/critic reference ABI, recurrent state, and
  numeric precision;
- residual scale, deterministic/stochastic semantics, addition, clipping, and
  normalized-to-physical action transform;
- tracking reward, admitted reference bytes, cadence, root frame, joint order,
  and Tier-D receipts;
- MuJoCo model, Gymnasium environment, observations, actions, dynamics, resets,
  termination, and wrapper topology; and
- trainer, hyperparameters, seeds, budget, checkpoint rule, protected
  evaluator, and software fingerprint.

Experiment 002B may change only the controller-visible numeric reference window
at matched states. A later Family-A comparison may change only the oracle and
its emitted mode, phase, window, transition reason, and next oracle state.

## Risks retained

- The pinned Farama runner constructs default `Humanoid-v5`, whose default is
  `terminate_when_unhealthy=true`; the local calibration uses `false`. The
  latter is an exact-project resource probe, not reproduction of Farama's
  learning result.
- A same-runtime rollout is dynamically feasible under its generating
  controller and state. It does not establish naturalness, robustness, branch
  coverage, or trackability from other resets.
- Additive residual control can saturate. Record base action, residual,
  unclipped sum, executed action, and lost residual authority separately.
- RSI and early termination are tracker-training choices. If adopted, bind
  them before admission and hold them fixed in later comparisons.

## Primary sources

- [Farama Humanoid expert description: TQC, 20 million steps, non-falling policy](https://github.com/Farama-Foundation/minari-dataset-generation-scripts/blob/e74f9d0524c6df014c5a9985d0804001b9ce40dc/scripts/mujoco/descriptions/Humanoid-expert.md)
- [Farama pinned MuJoCo training script](https://github.com/Farama-Foundation/minari-dataset-generation-scripts/blob/e74f9d0524c6df014c5a9985d0804001b9ce40dc/scripts/mujoco/train.py)
- [Farama pinned environment construction](https://github.com/Farama-Foundation/minari-dataset-generation-scripts/blob/e74f9d0524c6df014c5a9985d0804001b9ce40dc/scripts/mujoco/make_env.py)
- [SB3-Contrib TQC actor, tag `v2.9.0`](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/v2.9.0/sb3_contrib/tqc/policies.py)
- [TQC, arXiv `2005.04269v1`](https://arxiv.org/abs/2005.04269v1)
- [DeepMimic, arXiv `1804.02717v3`](https://arxiv.org/abs/1804.02717v3)
- [AMP, arXiv `2104.02180v2`](https://arxiv.org/abs/2104.02180v2)
- [Residual Reinforcement Learning, arXiv `1812.03201v2`](https://arxiv.org/abs/1812.03201v2)
- [DAgger, arXiv `1011.0686v3`](https://arxiv.org/abs/1011.0686v3)
- [PHC, arXiv `2305.06456v3`](https://arxiv.org/abs/2305.06456v3)
- [UniTracker, arXiv `2507.07356v3`](https://arxiv.org/abs/2507.07356v3)
- [ResMimic, arXiv `2510.05070v2`](https://arxiv.org/abs/2510.05070v2)
