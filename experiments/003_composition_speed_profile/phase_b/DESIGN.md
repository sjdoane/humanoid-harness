# Experiment 003 Phase B: fine-tuning runtime

| status | current truth |
|---|---|
| progress | FT2R3 closes the remaining FT2 isolation findings with sealed worker inputs and modules, a clean worker environment, bounded exact-schema IPC, parent-observed resource gates, fail-closed RSI evidence, and detector-path mechanism negatives. No training ran. |
| bottleneck | FT2R1 through FT2R3 still need one combined scientific and robustness re-review. Production-load resource behavior, calibration, smoke, cohort, protected utility, and humanoid behavior remain unmeasured. |
| next step | Fable reviews and commits FT2R3, then launches the combined re-reviews. Do not run the smoke before both verdicts; calibration remains required for scoring task success. |

## Frozen artifacts

| artifact | SHA-256 | role |
|---|---|---|
| `oracle_cycle_1_reference_v1.json` | `4d24f22780360d7632235572d97b3fccfe7c376162e69e15082f77bd7afcccf1` | `humanoid_reference_composition_oracle/v1`; Phase A guard grammar plus the fixed phase policy |
| `tracking_only_v1.json` | `eea2b6a9893e6e4ca5aea5d6787580f12062084e758cc9c27db2f1376bcb1c5f` | sole registered reward; `r_task = +0.0`; certified task-input adapter bound |
| `training_design_v1.json` | `1d104a52238eee4f5065b4fcc30c285662e26ab06edd847da7ca9e160364c69e` | frozen worker design; FT2 implements it but does not execute a training run |
| `utility_evaluation_design_v1.json` | `4145c7bae5deacd9156070aa6531dafc0e4c9041e4fac6a8e4033e0713334f7d` | bounded utility gate and disjoint calibration procedure |
| `starting_checkpoint_v1.json` | `9931750c5015ae5a385ac42202485868b0338530918421234c5aacbfe9b3e3b5` | exact expert, fresh E1 receipt, unchanged step-0 export, and value-network binding |
| `run_manifest_training_admission_v2.json` | `45ce5baa66b6d6dadba38c61bbdfc943a75c542ae7c355927f1031fc65661ebb` | reviewed pre-construction seal for smoke and cohort profiles |
| `receipts/e1_full_authority_warm_start_v1.json` | `5754db8e8afdc7f05f8a67ab3fb6a69ae453968a6c14e9f1dca545d688834bc9` | live-source-bound replay of four synthetic plus 64 SHA-ranked real fixtures and fresh optimizer state |
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
| CandidateTaskInputsV2 | adapter-certified built-in float from the stock body-mass-weighted COM-x delta, inclusive `[-25,25] m/s`, and exact `0.015 s` cadence; Astra's reviewed `task_inputs_v2.py` bytes remain unchanged |
| Registry | exact schema, formula, parser, bounds, and compositor hashes; `tracking_only/v1` remains the baseline and the reviewed F2 T2 adapter is registered for later bounded reward candidates |
| Report | `humanoid_composition_cycle_report/v2`, a backward-shaped report-v1 superset with separate input, policy, training, reference, reward, episode, summary, and integrity evidence |

## Claim ceiling

`exploratory_reference_conditioned_fine_tuning_utility_only_no_causal_reference_use_oracle_improvement_reward_improvement_generalization_naturalness_or_humanoid_competence_claim`

FT2 evidence remains `interface_check`. The implementation contains a training
loop, but no disposable smoke, cohort, checkpoint selection, utility evaluation,
or reward candidate ran.

## FT2 implementation

| surface | implemented boundary |
|---|---|
| Worker | CPU PPO with the frozen recipe, tanh-corrected likelihood, four `DummyVecEnv` environments, exact final-transition counting, eight-rollout reference-column/value-only stage, and full-actor unfreeze thereafter |
| Streams | Environment indices `0,1` are composition and `2,3` are rehearsal, giving exact `50/50` counted transitions; the shared SHA-ranked RSI scheduler covers the 27 block-origin cells and executes uncounted predecessor restoration |
| Supervision | Source/runtime inspection, FT1 and E003R1 manifest bindings, authoritative mailbox acceptance plus acknowledgment, exact canonical argv, reservation-derived deadlines, two-phase construction ACK, serial seeds, sealed inputs and executed modules, bounded JSON IPC, clean worker environment, CPU/wall/process-tree-RSS/disk/aggregate-output/throughput gates, process-group cleanup, and one immutable success or failure receipt per seed |
| Persistence | One final full checkpoint and strict actor export per successful seed; fixed member sets and order, manual NPY-v1 schema checks before allocation, per-member and total expansion caps, no-overwrite publication, frozen-fixture bitwise inference, checkpoint/export equivalence, and a production-only five-seed cohort index |
| Report v2 | Deterministic scientific receipt chained to E003R1; manifest- and RSI-byte-bound training-facts validation; explicit missing evidence; side-by-side trained and step-zero summaries; reload-time summary recomputation; separate host/wall/resource telemetry |
| CLI | `train` performs complete preflight before construction; `evaluate-policy` first validates all five cohort chains and executed evaluator sources, then runs the three hold cells and fixed round trip for a stored checkpoint and the step-0 actor under bounded worker supervision |

Training admission replays E1 and semantically parses the task, library,
reference corpus, evaluator, training design, oracle, reward, and starting
checkpoint before model or environment construction. The reviewed manifest
admits only smoke seed `121901` at `196,608` transitions or cohort seeds
`121001,121101,121201,121301,121401` at `1,048,576` transitions each, with
`final_transition_only` checkpoint selection.

Task-success thresholds are not placeholders. The calibration writer requires
the exact Cartesian split of blocks `120201`-`120220` and policy seeds
`122001,122101,122201,122301,122401`, disjoint from training and evaluation.
It freezes higher empirical 95th-percentile segment bands, two censored
transition-latency caps, and a settled-state band with declared margins. The
evaluator validates a supplied receipt before loading a checkpoint. Without a
frozen calibration receipt, task-success successes, total, proportion, and
interval are all `null` and explicitly non-scoring. With one, an episode
succeeds only when safety, all three segment tolerances, and both transition
latency bounds pass. Reports retain a 20-episode exact binomial interval per
checkpoint and paired candidate-minus-step-zero effects by PPO seed; they never
pool episodes across checkpoints or report pooled `n = 100` task success.

## FT2R2 integrity boundary

| finding | implemented repair |
|---|---|
| FT2-SCI-01 | Evaluator-owned pure calculations consume direct simulator state and exact reference rows. A canonical sufficient trace binds state fields, reference-row identity, and actions; every protected metric is recomputed from that trace without importing reward helpers. |
| FT2-SCI-02, ADV-05 | `evaluate-policy` validates the complete five-checkpoint production chain, current preflight, execution manifest, RSI ledger, receipts, and actual evaluator source realpaths and digests before output or model/environment construction. A bounded worker accounts atomically for all 160 planned episodes. |
| FT2-SCI-03 | Missing calibration is explicit non-scoring evidence. Calibrated success is conjunctive over safety, three segment tolerances, and two latency bounds, with checkpoint-level exact intervals and PPO-seed-paired effects only. |
| FT2-SCI-04 | Every PPO update first audits sampled rollout `old_log_prob` values against the stored rollout-time distribution snapshot. The receipt binds sample identity and snapshot; staged unfreeze and truncation/termination GAE semantics have positive and negative regressions. |
| FT2-SCI-05, ADV-08 | Report v2 validates manifest- and ledger-bound training facts, labels absent evidence as missing, preserves the interface-only claim, shows trained and step-zero summaries side by side, and recomputes both report and production-cohort summaries on reload. |
| FT2-SCI-06, ADV-01 | A reservation is accepted only through a digest-bound native mailbox acceptance record and acknowledgment. Owner, canonical argv, mode, commit, input hashes, output, proposal, authorizer, validity window, and derived deadline are checked before output creation and again immediately before spawn. |
| FT2-SCI-08, ADV-06 | Actor and full-checkpoint loaders accept fixed member sets and order only, validate NPY-v1 headers manually before allocation, and enforce member-count, per-member, and aggregate expansion caps with bounded streaming. |

The F2 registry entry binds
`target_speed_triangular_affine_t2_adapter/v1` to evaluator SHA-256
`064393887bb4a7157d614981cc2000940252e12c31ad0aabc13109737fd94301`,
`target_speed_triangular_affine_recipe/v1` and
`parse_target_speed_formula_recipe` to parser SHA-256
`c60dea03e6f0e71b81875fea59c84bd8fe00ce39f94ac7c54ca6e2a57360dccb`,
the exact 137-byte canonical bounds object to SHA-256
`2c6030264231a52320a44fe0f4d4ed519bc2e635b892f1b9c0d8929166106933`,
task-input source SHA-256
`9607f2d56a54922eac06ec7fcc740b79e68d4ed9e88b192602aae2900fb3f0d2`,
output envelope `[-10, 15]`, compositor
`tracking_plus_task_stock_telemetry/v1`, and mandatory consumption of the FT2R1
velocity-admission certificate. Candidate specifications carry flat `alpha`
and `beta` parameters within those bounds and evidence class
`exploratory_fine_tuning_cycle`; registering F2 is not evidence that a candidate
ran.

Handoff edits are committed in Fable's integration commits by design. FT2R2
therefore changes the handoff record but makes no code change for FT2-SCI-09
beyond documenting that integration boundary.

## FT2R3 isolation boundary

| finding | implemented repair |
|---|---|
| FT2-ADV-02 | Execution manifest v3 carries expected byte count and SHA-256 for 102 checkout-local inputs: oracle, reward, task, library, evaluator, training design, reference-corpus authorities and all 27 scheduled bundles, plus starting-checkpoint authorities. The worker verifies the full lineage before both construction admissions, at each real artifact use, before persistence, and after persistence. Every loaded `oracle_composition` module must resolve beneath the declared checkout and match the recorded source digest. |
| FT2-ADV-03 | Spawn uses an explicit environment allowlist and fixed single-thread variables. The worker applies `RLIMIT_CPU` where supported and records `unsupported` otherwise. The parent observes the worker process tree rather than trusting child RSS, monitors free disk and total job output, and derives CPU, seed, cohort, and job limits from the accepted hard deadline. Filesystem, process-tree, and process-group controls remain explicitly OS-best-effort. |
| FT2-ADV-04 | Parent and worker exchange canonical JSON with an exact frame and payload schema through a 4 MiB `send_bytes`/`recv_bytes` bound. Keys, JSON types, nesting, container and string sizes, artifact filenames, numeric finiteness, and completion bindings are checked before use. Malformed input has its own status and cannot reach construction or persistence. |
| FT2-ADV-07 | RSI evidence requires the ledger and reset-count APIs for all four environments, an entry for every rehearsal reset in environments 2 and 3, no entries elsewhere, contiguous global indices, and exact scheduler assignments. Missing, `None`, omitted, empty, or cell-drifted evidence is `counter_drift` before persistence. |
| FT2-ADV-09, FT2-SCI-07 | Training retains a primary exception when environment close also fails and records both outcomes. Injected clock, process, channel, RSS, disk, and output observers cross the real detectors. Production RSI shortcut negatives call `restore_predecessor_rsi`; cohort negatives exercise `3/5`, missing-checkpoint, and hidden-aggregate failures. |

The execution manifest is `humanoid_phase_b_execution_manifest/v3`; per-seed
success and failure receipts are v2. Success receipts retain the control
outcomes and the start/final executed-module identity digests. These are
software isolation and interface records, not evidence that production macOS
containment, throughput, memory use, training, or humanoid behavior succeeded.

The fake runtime exercises the real worker process, supervisor, persistence,
report writer, and CLI. It is controlled software evidence, not simulator or
policy-performance evidence. The real checks are limited to the FT1 200-step
step-0 path and one certified predecessor reconstruction followed by one
counted step.

## Commands (documented, not executed by FT2R3)

Disposable interface-check smoke:

```bash
python -m oracle_composition.harness.cycle_cli train \
  --experiment experiments/003_composition_speed_profile \
  --cycle 3 \
  --oracle experiments/003_composition_speed_profile/phase_b/oracle_cycle_1_reference_v1.json \
  --reward experiments/003_composition_speed_profile/phase_b/tracking_only_v1.json \
  --output /absolute/fresh/path/phase_b_smoke_121901 \
  --seeds 121901 \
  --transitions 196608 \
  --reservation /absolute/path/to/accepted_smoke_reservation.json \
  --expected-wall-seconds 180 \
  --smoke
```

The smoke is always `interface_check` and non-promotable. `--smoke --promote`
fails before preflight, model construction, or environment construction.

Five-seed cohort, only after both reservation and Samuel authorization:

```bash
python -m oracle_composition.harness.cycle_cli train \
  --experiment experiments/003_composition_speed_profile \
  --cycle 3 \
  --oracle experiments/003_composition_speed_profile/phase_b/oracle_cycle_1_reference_v1.json \
  --reward experiments/003_composition_speed_profile/phase_b/tracking_only_v1.json \
  --output /absolute/fresh/path/phase_b_cohort \
  --seeds 121001,121101,121201,121301,121401 \
  --transitions 1048576 \
  --reservation /absolute/path/to/accepted_cohort_reservation.json \
  --expected-wall-seconds 6360
```

Evaluate one stored final checkpoint and its step-0 comparator:

```bash
python -m oracle_composition.harness.cycle_cli evaluate-policy \
  --experiment experiments/003_composition_speed_profile \
  --cycle 3 \
  --oracle experiments/003_composition_speed_profile/phase_b/oracle_cycle_1_reference_v1.json \
  --reward experiments/003_composition_speed_profile/phase_b/tracking_only_v1.json \
  --checkpoint /absolute/path/to/checkpoint_seed_121001_final.npz \
  --calibration-receipt /absolute/path/to/frozen_task_success_calibration.json \
  --calibration-receipt-sha256 <reviewed-sha256> \
  --output /absolute/fresh/path/phase_b_utility_121001 \
  --reservation /absolute/path/to/accepted_evaluation_reservation.json
```

## Compute reservation

The full proposal text from survey section 10 is retained verbatim:

```text
owner=<owner>
command=<exact train command>
commit=<clean builder commit>
inputs=<oracle/reward/task/library/checkpoint/design hash ledger>
work=5 seeds; 5,242,880 counted transitions; serial seeds; 4 DummyVecEnv
expected_wall=106m
hard_wall=120m; per_seed=22m
rss_hard=8GiB
disk_free_preflight=20GiB
output_cap=8GiB
throughput_floor=800 steps/s after 65,536 transitions
output=<fresh absolute path>
conflict_check=no other heavy repository job
authorization=<accepted proposal id plus Samuel authorization id>
```

The CLI consumes reservation schema v2, whose digest binds the authoritative
native-mailbox acceptance message and its acknowledgment. It compares owner,
exact canonical argv, mode, clean commit, source/input hashes, fresh absolute
output, proposal ID, and required authorizer field by field; rejects stale or
unrecognized records; and repeats conflict and validity checks immediately
before worker spawn. The accepted reservation supplies the enforced deadlines:
the smoke uses `work=1 seed; 196,608 counted transitions; 4 DummyVecEnv`,
`expected_wall=3m`, and a hard total and per-seed wall of `20m`, never the
cohort's `22m` per-seed default. All other fixed resource declarations remain
unchanged. Production training also requires the exact positive
`--expected-wall-seconds` used to reserve the shared token. Training and
evaluation validate that complete token against the already admitted
reservation immediately before each worker spawn, retain its owner/token ID
through process cleanup, and release only that exact pair on success or
failure. Evaluation's fixed supervisor wall is 1,800 seconds and must be the
expected wall supplied when its token is reserved.
