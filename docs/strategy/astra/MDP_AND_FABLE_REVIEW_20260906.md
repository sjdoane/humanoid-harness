# MDP choice and independent critique

| progress | Source review favors testing a competent supplied tracker, not protecting native sunk cost. |
|---|---|
| bottleneck | Neither native Phase B competence nor a safe local GMT adapter is demonstrated. |
| next step | One capped native smoke; static GMT admission review in parallel. |

## Decision

| route | useful starting point | unmeasured cost / boundary |
|---|---|---|
| Native Humanoid-v5 | Existing expert initialization, 4-environment PPO, strict reference interface | Numeric-reference competence is unproven; torque-policy adaptation may be expensive |
| GMT / G1 | Pretrained motion tracker, known PD interface, Mac deployment and supplied motions | Safe weight/data conversion, exact wrapper, task-reward training and novel transitions |
| HumEnv / Meta Motivo | Safe-format weights and shared motion/reward latent interface | Latent prompting is not PPO fine-tuning; robot/data licenses and reference conversion differ |
| SONIC | Strong reference-conditioned whole-body controller | Released training route adds Linux/NVIDIA dependencies |

- Keep the native attempt at 196,608 transitions and 1,200 s; a budget is not a
  throughput measurement or a promise of reference competence.
- Prefer GMT if admitted supplied-motion competence, composition and controllable
  task adaptation make it the shorter route to a qualifying demonstration.
- Start a GMT family with a frozen low-level tracker. A trainable high-level
  policy is plausible; whether it can affect the task while respecting the
  authored reference is an experiment, not a conclusion.
- No arbitrary percentage threshold or release-relative tracking threshold is
  adopted without a defined metric, baseline and matched protocol.

## Fable review receipt and disposition

- Read-only CLI review completed: session
  `b5e4f1e3-347a-4eb5-b2d0-b9b653b5ad6b`.
- Client result reports `claude-fable-5-1`, requested max effort, no spawned
  agents. This is client-reported model metadata, not independent server attestation.
- Project Read/Glob hooks denied access because SessionStart supplied no model
  field. No files were read. The review is prompt-based advice, **not code review**.
- The active long-lived Fable session remains separate; role-change messages
  are queued there until its current builder checkpoint. Do not claim acceptance.

| Fable suggestion | Astra disposition |
|---|---|
| Frozen tracker with higher-level learning | Worth testing as a separately frozen adapter family |
| Prevent indirect held-out leakage through history | Adopt: candidate prompts exclude protected-derived routing, retrieval and steering |
| PD targets may help tracking | Plausible new action-space hypothesis, not a demonstrated order-of-magnitude gain |
| Native throughput is 164 steps/s | Reject as measurement: 196,608 / 1,200 is the required average, not observed performance |
| Native is unvectorized / starts from scratch | Correct: the existing recipe has 4 environments and expert actor initialization |
| Seven splice survivals prove the option MDP works | Reject: arbitrary cutoff; transition failure alone does not identify residual learning as the necessary repair |
| Reduce routine review round trips | Adopt: machine checks for fixed contracts; review on substantive changes or unexpected outcomes |

The CLI reported about 100,801 cache-creation input tokens for this bounded
review. Project context overhead is too high for frequent reviews. Use compact
review packets and fix supported read-only identity handling before repeating;
do not disable provider safeguards or fabricate identity events.

## Primary sources

- [GMT source and deployment](https://github.com/zixuan417/humanoid-general-motion-tracking)
  and [paper](https://arxiv.org/html/2506.14770): supplied tracker and interface,
  not our proposed fine-tuning result.
- [Meta Motivo](https://github.com/facebookresearch/metamotivo) and
  [HumEnv](https://github.com/facebookresearch/humenv): alternate latent-conditioned
  policy and simulation family.
- [SONIC source](https://github.com/NVlabs/GR00T-WholeBodyControl): alternative
  whole-body tracker; not currently a local training integration.
- [PyTorch TorchScript load warning](https://docs.pytorch.org/docs/2.8/generated/torch.jit.load.html):
  downloaded executable model archives are not data-only admissions.
