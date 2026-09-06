# Reference library: local setup and compatibility

| status | current truth |
|---|---|
| progress | Exact native corpus and bound actor/receipt files are staged in Astra; 36 training clips decode through the real reference loader. |
| bottleneck | Data compatibility is not trained tracking. Runtime reviews, calibration and causal-reference evidence remain open. |
| next step | Finish the bounded existing T1 tracker admission/smoke before a new joint speed-zone task. |

## What was actually set up

- Source: Fable's retained `artifacts/reference_corpus_v2/`, named in the committed handoff at main `ed9f1d38aba4f7b41a576b0fd9c8be4f6b8b47fe`.
- Destination: `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra/artifacts/reference_corpus_v2/`.
- Copy: full tree, no symlinks; source and staged copy matched by per-file SHA-256 before promotion into the destination.
- Contents: **477 regular files, 128 NPZ payloads, 1,077,406,602 bytes**.
- The 128 payloads are not 128 admitted training clips. The manifest has 108 corpus clips; other retained objects include development-screen evidence.
- Ten additional exact files were copied from paths bound by the committed library/starting-checkpoint manifests: three actor NPZs, their six import/equivalence receipts, and the Step-0 full-authority actor export.
- Existing relative paths now resolve in Astra. No new loader, environment, trainer, policy code, task configuration or dependency was installed for this setup.
- Payloads remain Git-ignored and local-only. The upstream Minari controller/data license is unresolved; no redistribution permission is inferred.

## Verification receipt

| check | result / identity |
|---|---|
| Whole-tree content comparison | exact match before promotion; source untouched |
| Corpus manifest | `a3e9a7234b67194f0d4ac8d3961e9040f217ec9768e27451c279105aa3391090` |
| Corpus index | `9347c741241b1138ce361f01ad319d05e09a7597c9ef51d9715b86dee6c34afa` |
| E3 certificate | `5a91a265e057e6f8c6497d50a4a20ac3baab40b67c101115bc96b6f4b7555b74` |
| Tier-D aggregate | `dcf06c8e96ebd4d7c3171ab6d3062a73045addcb1be992e0e2d3ca5f62801bf7` |
| Certifier request | `cc2d70800c59828716950365da678f85c2b04422d797ee859e828672c94acac4` |
| Library artifact bindings | `verify_library_artifacts` passed for exact corpus, actor and receipt byte bindings |
| Numeric-loader check | `load_v2_reference_clip`: seeds 120001–120012 × expert/medium/simple; **36/36 loaded** |
| Decoded layout | 1001 × 45 reference rows; 1001 × 348 boundary observations per checked clip |
| Split handling | Only training clips decoded; held-out payloads copied/hashed opaquely, not used to choose behavior |
| Simulator | Not imported by the numeric-loader check; no environment constructed or stepped |

These checks establish retained bytes, manifest links and readable numeric
format. They do **not** reissue a dynamics certificate, admit every training
block, validate current source-bound runtime receipts, or demonstrate tracking.
Failed blocks remain failed. The loader check is not an E3 admission decision.

Exact training/run bindings remain in:

- `experiments/reference_corpus_v1/e3_manifest_v1.json`
- `experiments/003_composition_speed_profile/library_manifest_v1.json`
- `experiments/003_composition_speed_profile/execution_manifest_v1.json`
- `experiments/003_composition_speed_profile/phase_b/run_manifest_training_admission_v2.json`
- `experiments/003_composition_speed_profile/phase_b/starting_checkpoint_v1.json`

The final two files are existing contracts, not a claim of current launch
approval. Source-bound receipts can become stale after a source repair.

## Which references fit?

| Library | Interface / evidence | Decision |
|---|---|---|
| Native v2 corpus | Humanoid-v5, 17 actions, 15 ms; 45-value reference rows, 8-frame window | Use first; same adapter as existing research |
| Existing DeepMimic four-clip subset | Different skeleton; 44-value frames, mixed cadence; MIT repository license | Retain as source candidates, not native targets. Three pass source-format audit; face-down get-up does not |
| [GMT, pinned `2a590de`](https://github.com/zixuan417/humanoid-general-motion-tracking/tree/2a590de25a1eb08e47491977a738549c22f16e1f) | G1, 23 actions, 20 ms; pickle motions and TorchScript model; authors report M1 macOS testing | Useful separate tracker pilot, but requires safe conversion/review and a new adapter. Not installed |
| [SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl) | G1 tracker with planner and public checkpoints; different reference/controller interface and GPU-oriented stack | Strong future supplied-controller option; not a drop-in Mac Humanoid-v5 configuration |
| [LocoMuJoCo](https://github.com/robfiras/loco-mujoco) | Robot-specific data and MDPs; [dataset terms](https://huggingface.co/datasets/robfiras/loco-mujoco-datasets) differ from code license | Do not migrate merely to obtain motions; resolve rights and adapter mapping first |
| [HumEnv](https://github.com/facebookresearch/humenv) / [Meta Motivo](https://github.com/facebookresearch/metamotivo) | Different SMPL-based embodiment/action space; pretrained models and separate data rights | A system-level comparator, not the current frozen-family baseline |

The old RL-Sculptor DeepMimic reader extracts root/clock features; it is not a
full joint retargeter or a compatible pretrained tracker. Converting that source
library is more work than using the native corpus already available.

## Visual task sequence

1. **Mechanism gate:** existing T1 expert → simple → expert numeric schedule,
   tracking-only reward, one controller family; exact/zero/shuffled/time-shifted
   reference controls. A video of complete-controller switching does not pass.
2. **Reward gate:** freeze an admitted oracle and test the separate task-reward
   intervention with independent speed and fall metrics.
3. **Joint task:** new versioned straight-course speed zones; compare the four
   oracle × reward arms and a handwritten composer. See the first-principles note.

- The existing simple gait has more recorded E3 falls than medium. If T1 fails,
  preserve the result; propose a separate expert/medium/expert study rather
  than replacing the failed condition inside T1.
- The medium gait is close to 3 m/s already. Do not use that as the fixed oracle
  for a 3 m/s reward study and present trivial headroom as successful adaptation.
- A single exploratory video is a pipeline demonstration, not robustness or
  generalization evidence. Show synchronized phase/speed/failure traces and
  disclose every controller/reference/reward actually used.
- Heavy simulation, training and any cohort still require their existing
  reviewed resource and authorization gates. No new run was launched here.
