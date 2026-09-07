# Humanoid Harness

An LLM-guided research harness with two design knobs: reference composition
and task reward. Training and objective evaluation remain separate from authoring.

```text
INPUT: task + admitted references + verified development evidence
                              |
                  LLM proposes oracle OR task reward
                              |
                    data-only candidate admission
                              |
state -> oracle -> reference window -> frozen base actor + learned residual
  ^                                                              |
  +------------------------- fixed G1 plant <---------------------+
                              |
                 OUTPUT: recorded states + task metrics
                              |
             independent evaluator -> feedback -> next LLM proposal

TRAINING: fixed tracking reward + authored task reward -> PPO -> residual
FROZEN: scene/reset, observations/actions, base weights, trainer/budget,
        tracking reward, evaluator, and final-checkpoint selection
```

## Current state

| Boundary | Evidence |
|---|---|
| Research target | Steerable, evidence-driven iteration over reference-oracle logic and task reward |
| Active implementation | GMT/G1 through Gymnasium; state-triggered motion selection; bounded residual PPO; separate tracking/task rewards; recorded-state feedback; immutable candidate admission |
| Measured development result | Walk → crouch → walk survives 20 seconds after training in three of three seeds with the initial task reward; tracking-only training survives one of three. No full posture-course gate pass. |
| Real revision loop | Retained LLM reward and oracle proposals were admitted, trained, evaluated, and rejected when their fixed criteria failed. Negative outcomes are not task success. |
| Reference use | Two five-arm closed-loop tests change actions and trajectories; exact arms reproduce their retained controls. This is input dependence, not composition quality. |
| Limits | One flat-ground course and fixed start. Within-clip playback still uses a local clock. No held-out generalization, obstacle clearance, locomanipulation, or foolproof-system claim. |

See the [measured results and replay receipts](docs/strategy/astra/G1_LEARNING_RESULTS_20260906.md)
and [current handoff](docs/operations/dual-orchestration/ASTRA_HANDOFF.md).
The blue region in the G1 replay is a posture constraint, **not a physical obstacle**.

## Quick start

Python 3.12–3.13 and `uv`:

```bash
uv sync --locked --extra sources --extra gym --extra train --extra dev
uv run humanoid-harness doctor
uv run humanoid-harness research build
uv run humanoid-harness research query "phase recovery"
uv run humanoid-harness g1 --help
```

- Motion/controller downloads are not bundled. Use the
  [GMT admission and baseline record](docs/strategy/astra/GMT_BASELINE_20260906.md);
  do not execute downloaded pickle, TorchScript, or upstream Python.
- `g1 feedback` verifies a pinned run and recomputes boundary metrics before
  producing proposal-safe feedback.
- Training uses the exact-source, resource-bounded launcher documented in the
  [G1 pilot](docs/strategy/astra/G1_COURSE_PILOT_20260906.md).
- `humanoid-harness ui --help` documents the read-only evidence UI and its
  explicitly registered G1 results. The local development instance is on port
  8766; linked reports retain studies outside the selected UI registry.

## Research ownership

- Astra leads strategy, implementation, integration, and bounded local training.
- Fable supplies independent reviews, feedback, and ideas; no parallel build lane.
- Sol workers receive small, isolated implementation or review tasks.
- Model requests and sender labels are not served-model attestation.
- [Operating authority](docs/strategy/astra/SYSTEM_LEAD_PLAN_20260906.md) ·
  [coordination protocol](docs/operations/dual-orchestration/README.md)

## Test and experiment boundaries

- [Test matrix](docs/operations/TEST_MATRIX_20260906.md): focused G1 checks and
  whole-repository failures are reported separately. The full suite is not green.
- Native `Humanoid-v5` is a separate, earlier experiment family, not a matched
  control for G1. Its source, failures, and sealed historical results remain:
  [Experiment 003](experiments/003_composition_speed_profile/PROTOCOL.md),
  [Experiment 004](experiments/004_t2_reward_study/PROTOCOL.md).
- An accepted artifact, successful launch, or video is not a successful task.
- Never change evaluator thresholds, tracking, or the MDP to rescue a candidate.
- Development evidence may guide revision. Protected evaluation may not.
- PRAXIST is optional orchestration, not a source of scientific truth.

## Repository map

```text
src/oracle_composition/
  adapters/gmt/   G1 plant, composition, residual training, rewards, replay
  feedback/       trace verification and development diagnosis
  harness/        contracts, execution, evaluation, resource coordination
  research/       literature index and retrieval
  sources/        data-only source admission
  ui/             read-only research and registered G1 evidence
docs/             contracts, strategy, runbooks, measured results
experiments/      earlier frozen study definitions and receipts
tests/            positive, negative, and integration checks
```

[Project charter](docs/PROJECT_CHARTER.md) ·
[Scientific boundary](docs/SCIENTIFIC_BOUNDARY.md) ·
[Research-source review](docs/ENGINEERING_TOOL_REVIEW.md)

No project license has been selected. Imported sources retain their own
attribution and restrictions; local admission does not grant redistribution rights.
