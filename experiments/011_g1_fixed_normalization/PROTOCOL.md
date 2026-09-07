# Fixed residual-input normalization: one matched learning screen

| progress | The reference-input gate is closed; the prior low-rate O2r1 run and all artifacts are retained. |
|---|---|
| bottleneck | The residual learner has not passed the full task. Raw inputs and reward incentives are separate possible contributors. |
| next step | Review the fixed-normalizer implementation; reproduce the raw-input control, then train one normalized candidate. |

## Question and factor

- Does a fixed transform of the residual learner's base-observation block
  improve training under the existing O2 oracle and r1 task reward?
- Only factor: raw first 2,154 input dimensions versus the pinned GMT actor's
  `(observation - mean) / (std + 1e-4)` transform. No clipping or online fitting.
- Keep the 11 task features and six oracle features raw and unchanged.
- Implement inside a non-trainable policy feature extractor consumed by both
  training and prediction. Do not change environment observations or base actor.
- This is trainer conditioning, not an oracle/reward innovation or an MDP win.
  The normalized coordinates may help, hurt, or have no useful effect.

## Frozen comparison

| Item | Exact setting |
|---|---|
| Oracle / reward | Existing O2 / r1; no edits |
| Seed | 20260906 |
| Transitions | 131,072 per arm; fresh initialization; final checkpoint only |
| PPO | Existing profile 2: learning rate 3e-5, total reward scale 1/64; all other settings unchanged |
| Candidate | New profile 3, identical PPO plus the fixed transform |
| MDP / evaluator | Existing legacy 2,171-observation G1 course; unchanged task gates |
| Order | Raw control reproduction, then normalized candidate |
| Resources | One accepted local job at a time, existing per-job 1,200 s / 8 GiB limits |

- Retained control: sibling
  `gmt_course_o2r1_lr3e5_131072_seed20260906_20260907/`.
- Control config: `2d8f13452ebc96d249173d4655b4153985bb9ac548dfbe6cc08566d563647469`.
- Control manifest: `3f2ea5f917b5136dc1483fb428848830f999fe2db7d2f1f8ed01933889a8158e`.
- Control resource: `92a748b0cdfc3b30c4deff989b26d52bd16a93c6fd66fe357daacca522181e20`.
- Candidate config and exact-source reservation must be pinned after independent
  implementation review and before any new training. This protocol alone does
  not authorize an unreviewed runtime or bypass resource exclusion.

## Implementation admission and stop rules

1. Effective contract pins slice `[0, 2154)`, float32 buffers, epsilon, zero
   trainable extractor parameters and exact mean/std byte identities:
   - mean: `72b2c94ae9873c573d480bca87d5914fc6e56e4677f07836c00dceb265b97ee6`;
   - std: `bc90eb2f9426d5aa226e4ef563d74533d2a6f8a00bc3ef228b47528f29efc765`.
2. Missing, altered, non-finite, wrong-shaped or nonpositive-std buffers fail.
   Numeric saved policies retain the transform; they are not optimizer resumes.
3. Same-seed common trainable tensors must match the raw control initially.
   The zeroed action-mean layer must produce exact zero through both real
   deterministic prediction paths. Artifact files differ because buffers differ;
   do not require an identity transform or claim full-file policy parity.
4. Reproduce all ten retained raw-control outputs byte-for-byte at the new
   source before running the candidate. A mismatch stops the study for diagnosis.
5. Candidate zero-residual trajectory, frames and evaluation must match the
   raw control exactly. The MDP/base controller did not change.
6. Every rollout records the maximum absolute normalized base input computed
   from actual rollout-buffer observations. Non-finite values invalidate the
   run; no arbitrary magnitude cutoff is presented as a physical safety gate.

## Predictions and advancement decision

All required for a one-seed development advance:

- Full 20 s with no fall, both oracle switches, observed spatial-region exit
  after entry, and at least 25 inside-region samples.
- All-visits posture compliance >= control's exact `0.6835443037974683`.
- Inside minimum root height <= control's exact `0.5151726255965129` m.
- Inside mean-speed target deviation <= fixed task threshold `0.10 m/s`.

Diagnostics, not substitute success criteria:

- Predicted residual RMS <=0.15 versus control's 0.206030; report even if false.
- Predicted last-quarter fall fraction and falls per transition no greater than
  the first quarter. Report both denominators, exposure and censored episodes.
- Report explained variance, KL-stopped updates, attempted epochs, normalized
  input range, reward components, all objective gates and final trajectory.
- The control falls at 5.76 s. Its shorter-horizon lateral maximum and aggregate
  errors are not fair full-horizon performance baselines. Report common-prefix
  diagnostics separately; candidate must still meet the unchanged task gates
  before claiming full task success.

No survival pass means reject. Survival with a failed advancement criterion
means retain but do not adopt. Passing this screen permits a separately pinned
replication on seeds 20260907 and 20260908; it does not change defaults or prove
generalization. No early checkpoint selection, extra seed, reward revision,
reference replacement or evaluator relaxation after seeing the result.

## Rival explanations and limits

- Reward incentives, reference mismatch and residual/base-history interaction
  remain rival explanations for poor learning. One transform cannot isolate them.
- Fixed-base reference ablation ranges do not characterize trained-residual
  distributions; actual training ranges are needed.
- The transform reuses existing controller statistics but operates before a
  separately learned residual MLP. That compatibility does not guarantee benefit.
- Preserve the full two-knob research target. This trainer study supports that
  target only if it enables a later trained, evaluated oracle/reward iteration.
