# Experiment 003 Phase B: fine-tuning runtime

| status | current truth |
|---|---|
| progress | FT1 implements and receipts the no-learning contracts, exact 708-D full-authority warm start, switch-only phase transfer, composed reference windows, and tracking-only reward composition. |
| bottleneck | No policy has been trained. These checks establish interfaces and step-0 identity only; they do not establish reference use, transition success, or humanoid competence. |
| next step | FT2 adds the reserved training worker, balanced supervision and RSI scheduler, persistence, report-v2 writer, and `cycle_cli train`; training still requires the stated reservation or authorization. |

## Frozen artifacts

| artifact | SHA-256 | role |
|---|---|---|
| `oracle_cycle_1_reference_v1.json` | `4d24f22780360d7632235572d97b3fccfe7c376162e69e15082f77bd7afcccf1` | `humanoid_reference_composition_oracle/v1`; Phase A guard grammar plus the fixed phase policy |
| `tracking_only_v1.json` | `aeb14853490aed4a9195a6efeb4910d1a670a1edcd843d2d322c43d6ce655b83` | sole registered reward; `r_task = +0.0` |
| `training_design_v1.json` | `1d104a52238eee4f5065b4fcc30c285662e26ab06edd847da7ca9e160364c69e` | prospective FT2 design; this slice does not execute it |
| `utility_evaluation_design_v1.json` | `0dbcd6c6991f200cfec72033238b3a9b067684e600ded5ed5311c8b76f7393e0` | bounded utility gate definition |
| `starting_checkpoint_v1.json` | `964868d069652766f8527d23eccc8b904906784a7792f564a292785da3f1fb28` | exact expert, E1 receipt, step-0 export, and fresh value-network binding |
| `run_manifest_interface_check_v1.json` | `28d800f077723382f892f3ca6e68c4b33d79e6a3e30478b7b4888fd2d64f9e0e` | complete pre-model-construction binding for the interface check |
| `receipts/e1_full_authority_warm_start_v1.json` | `c602e14ab4cea1b856eaf4cd86240035214d2af096381429ececac3f7e2f3788` | four synthetic plus 64 SHA-ranked real fixtures |
| `receipts/phase_transfer_static_v1.json` | `eee389c5dbb0fd030457c9022e879bb3f87794726f7c09446c8ffea91e9b7953` | nine-block static splice check |
| ignored step-0 actor export | `6ebc2b56be9a5f304b8b584157fd0141d449d75297366213e4976291cb2dcfe0` | local-only strict code-free actor export; not eligible for Git redistribution |

All JSON artifacts above use sorted, compact, finite canonical JSON bytes. Phase A v1 artifacts are inputs only and remain byte-identical.

## Full-authority step-0 policy

| field | frozen value |
|---|---|
| Source | strict expert NPZ `60987a4e054db2e04f9cb3ab73e13dfe8e2f3ec7dec46346d2b9d0277ad18d9b` |
| Input | C-order little-endian `float32[708] = state[348] || reference[8,45]`, row-major |
| Actor | `708-256-256-17`, ReLU, complete normalized action; no residual, blend, expert bypass, or normalizer |
| Transfer | expert `W_x`, first bias, second layer, mean head, and state-dependent log-std head copied bitwise; all reference-column bits are positive zero |
| Distribution | squashed diagonal Gaussian; state-dependent log-std clamped to `[-20, 2]`; tanh-corrected likelihood |
| Physical action | exact float32 `low + 0.5 * (u + 1) * (high - low)`; generic rescale wrappers refused |
| Value | fresh independent `708-256-256-1` network, seed `20260905`; no actor feature sharing |
| Initialization | `ortho_init=false`; optimizer authority check requires every actor and value parameter |
| Export | versioned deterministic NPZ retaining both mean and log-std heads; no executable code, value head, optimizer, normalizer, or RNG |

E1 passed on all `68` fixtures. Copied parameters, mean, log-std, deterministic and seed-`20260905` stochastic normalized/physical actions, export, and reload are bitwise equal. Independent PPO likelihood recomputation differs by at most `4.76837158203125e-6`, below `1e-5`.

## Reference runtime

| event | rule |
|---|---|
| Reset | initial behavior at phase `0` |
| Behavior change, recovery entry, rejoin | float64 nearest-state search over `0 <= j <= t` |
| Score | lexicographic `(maximum of six normalized errors, sum of their squares, j)` |
| Hold or same-behavior state transition | advance one boundary after each plant step; never rematch |
| Policy window | selected rows `j...j+7`, converted to row-major float32 only at the policy boundary |
| Reward target | exact float64 row `j+1`, hidden from candidate reward code |
| Wrap/end | never wrap; hold only the actual final reference row |
| Lookahead | none before a live guard fires |

The nine-block static receipt exactly reproduces the survey:

| transition | same-index maximum normalized error | nearest-phase error | selected phase |
|---|---:|---:|---:|
| expert boundary `300` to medium | `1.512-5.240` | `1.263-2.574` | `45-281` |
| medium after `300` steps to expert | `2.091-7.188` | `1.034-2.020` | `23-427` |

This is a reference-row comparison, not evidence that the simulator survives either transition.

## Reward and report boundary

| surface | frozen rule |
|---|---|
| Training reward | `r_train = r_track + r_task`; baseline `r_task = +0.0` |
| Stock Humanoid reward | named telemetry only; never added to training reward |
| CandidateTaskInputsV2 | trusted float64 COM forward velocity in `[-25,25]`, exact target `3.0`, exact cadence `0.015 s`; no other field or callback |
| Registry | exact schema, formula, parser, bounds, and compositor hashes; only `tracking_only/v1` admitted in FT1 |
| Report | `humanoid_composition_cycle_report/v2`, a backward-shaped report-v1 superset with separate input, policy, training, reference, reward, episode, summary, and integrity evidence |

## Claim ceiling

`exploratory_reference_conditioned_fine_tuning_utility_only_no_causal_reference_use_oracle_improvement_reward_improvement_generalization_naturalness_or_humanoid_competence_claim`

FT1 evidence class is `interface_check`. No training loop, cohort, checkpoint selection, or reward candidate ran.

## FT2 adds

- Parent/worker validation and reservation boundary.
- The four-environment `50/50` composition/rehearsal scheduler and certified RSI reconstruction.
- PPO updates, eight-rollout reference-column/value-only warm-up, then full-actor unfreeze.
- Atomic full checkpoints, final strict exports, per-seed receipts, and report-v2 writing.
- Fake-runtime CLI coverage and the separately authorized disposable smoke or cohort.
