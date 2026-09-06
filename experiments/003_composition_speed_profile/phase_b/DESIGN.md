# Experiment 003 Phase B: fine-tuning runtime

| status | current truth |
|---|---|
| progress | FT2R1 closes the FT1/E003R1 training-admission findings: live-source E1 replay, exact optimizer receipts, certified task inputs, a reviewed execution seal, bounded actor reload, trace-derived protected metrics, strict designer provenance, and an executable disjoint calibration contract. No training ran. |
| bottleneck | No calibration receipt, disposable smoke, cohort, or protected utility evaluation exists. Interface checks do not establish numeric-reference use, transition success, oracle or reward improvement, or humanoid competence. |
| next step | Fable reviews and commits FT2R1. Calibration data must be collected and frozen before task-success evaluation; any later smoke still requires its own accepted mailbox reservation. |

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
| Registry | exact schema, formula, parser, bounds, and compositor hashes; only `tracking_only/v1` admitted in FT1 |
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
| Supervision | Source/runtime inspection, FT1 and E003R1 manifest bindings, two-phase construction ACK, serial seeds, wall/RSS/disk/output/throughput gates, process-group cleanup, and one immutable success or failure receipt per seed |
| Persistence | One final full checkpoint and strict actor export per successful seed, no-overwrite publication, strict canonical reload, frozen-fixture bitwise inference, checkpoint/export equivalence, and a five-seed cohort index |
| Report v2 | Deterministic scientific receipt chained to E003R1; separate host/wall/resource telemetry; per-seed training facts; protected speed, tracking, safety, switch, resynchronization, task-success, and utility-gate fields |
| CLI | `train` performs complete preflight before construction; `evaluate-policy` runs the three hold cells and fixed round trip for a stored checkpoint and the step-0 actor on the same 20 blocks |

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
evaluator rejects a missing or mismatched receipt before loading a checkpoint.
Reports retain a 20-episode exact binomial interval per checkpoint and paired
candidate-minus-step-zero effects by PPO seed; they never report pooled
`n = 100` task success.

The fake runtime exercises the real worker process, supervisor, persistence,
report writer, and CLI. It is controlled software evidence, not simulator or
policy-performance evidence. The real checks are limited to the FT1 200-step
step-0 path and one certified predecessor reconstruction followed by one
counted step.

## Commands (documented, not executed by FT2)

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
  --reservation /absolute/path/to/accepted_cohort_reservation.json
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
  --output /absolute/fresh/path/phase_b_utility_121001
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

The CLI consumes a canonical JSON translation with separate `hard_wall` and
`per_seed_wall` fields, `schema_version: 1`, `accepted: true`, and
`proposal_id`. It verifies the exact source/input ledger and clean commit,
requires the output and work declaration to match the selected smoke or cohort,
and seals the accepted reservation into the execution manifest. The smoke uses
`work=1 seed; 196,608 counted transitions; 4 DummyVecEnv`,
`expected_wall=3m`, and `hard_wall=20m`; all other fixed resource declarations
remain unchanged.
