# Experiment 004: T2 expert-hold reward study

| status | current truth |
|---|---|
| progress | T2A freezes the expert-hold oracle, expert-only training design, reward-independent evaluator/report contract, matched-arm manifest, and uncalled pairing adapter; F3PINR1 adds the non-model-facing pre-dispatch seal. No candidate call, training, smoke, cohort, or behavioral evaluation ran. |
| bottleneck | Execution remains **NO-GO**: `t2_seal_v1.json` records `pairing_receipt: pending`; the candidate reward SHA-256 and an integrated `phase_b/training.py` pairing receipt are TBD. The adapter receipt does not show that the production trainer consumes the pairing key. |
| next step | Finish independent review of F3PINR1 and obtain Astra's accepted integrated pairing receipt. Only a later explicit dispatch verdict may permit the one canonical F3 call; smoke and cohorts remain separately gated. |

## Frozen artifact ledger

| artifact | bytes | SHA-256 | evidence |
|---|---:|---|---|
| `oracle_expert_hold_v1.json` | 964 | `489b82591cf65034d58d30f921442a84c93c9c98885cb2f5b22dcaf645a77010` | canonical Phase B oracle; no transitions or recovery |
| `training_design_t2_v1.json` | 1,815 | `84543f08265dae5076697549f25ce69b7e5e87947d4bb6e32b86a7700d24e67e` | exact T2 Phase B design contract |
| `evaluator_design_t2_v1.json` | 1,602 | `d8f54f80051e4e3e58a540fc1414e9a971c9e299d903c2a00162bfebaa64a04b` | evaluator/report/protected-core source-bound design |
| `t2_reward_study_expert_hold_v1.json` | 5,474 | `7aefaafcef6d319dfd991b6a16eccc5f5578651d6ac8ebb3573358cc307f0738` | common fields duplicated and equality-checked across arm slots |
| `t2_seal_v1.json` | 1,400 | `8c04f0c2bc9222f9918829f1e5d922c948583ea03da0eb0174a03e2dfcb314eb` | non-model-facing hash ledger; pairing receipt explicitly `pending`, dispatch withheld |
| `artifacts/experiments_004/t2_pairing_adapter_receipt_v1.json` | 1,736 | `01c591c566554c386fbfa38cdfc1019579e20af4a3c9a3e72dca958b09fed1f1` | fake-runtime adapter receipt; ignored local artifact, not trainer execution |
| `reward_study/pairing.py` | source | `ca90cd0451d3b413900c4de84a10cfea445a7159251f8639f548f1c64ae00943` | arm-invariant adapter source bound into both study arms and its receipt |
| `reward_study/t2_evaluator.py` | source | `880cd6980ba37eb6079d6bc97a5f0c6c8e9c9ff051e2439eaf1dbf8fe7d7ddb2` | protected direct-state computation |
| `reward_study/t2_report.py` | source | `f1dd474d5cd15e01a3a2e6a5b008d82f7f3630cc932b3dcec01cee50406e5cc7` | deterministic receipt and separate telemetry writer |
| `phase_b/protected_metrics.py` | source | `eca9269859266efa79cd16836cbe68abb63d7749d5681235306268f44cf9c493` | bound metric core used for COM and six tracking errors |
| `phase_b/report_v2.py` | source | `8a3c8d5cd0a02736759d30461b0a4e8cb4461534ac0b2d4c0386a23c4b96aea7` | bound deterministic-receipt and telemetry schema authority |

The common arm-invariant pairing key is
`4af9c9539bb2c7f9720f3e98a3f195c2b297d6bfe6adc51a1d63e7a8e4ad2464`.
It excludes only reward, arm label, output path, and timestamps. Changing the
common target from `3.0` to `3.1 m/s` changes the key to
`b683db801ec50d46a70110f9202cd6784d4dda51a87f7d44734dc55fa753b577`
and changes the derived fake-runtime stream receipt.

F3 preparation must verify the exact seal bytes plus every sealed file before
rendering the model prompt. The seal and its evaluator/study-manifest contents
are retained outside the 6,011-byte prompt. Because the integrated pairing
receipt is still TBD, this seal is a verified preparation input, not dispatch
authority.

## Frozen protocol `t2_reward_study_expert_hold/v1`

| field | frozen value |
|---|---|
| Family and authorable factor | Family B; the reward specification only |
| Task | Stock `Humanoid-v5`; expert start; target COM forward speed `3.0 m/s`; `1,000` steps |
| Oracle | `expert_hold/v1`; SHA-256 `489b82591cf65034d58d30f921442a84c93c9c98885cb2f5b22dcaf645a77010`; hold expert from authentic boundary `0`; no transitions or recovery; expert-reference RSI only. A `medium_hold/v1` route would be a new preregistered study. |
| Arms | `tracking_only/v1` SHA-256 `eea2b6a9893e6e4ca5aea5d6787580f12062084e758cc9c27db2f1376bcb1c5f` versus one admitted `target_speed_triangular_affine_t2_adapter/v1` specification SHA-256 **TBD after the single F3 call** |
| Candidate timing | The independent evaluator is frozen first; one initial F3 call; admit the candidate and seal both arm manifests before exposing baseline measurements |
| Common references and initialization | Corpus `a3e9a7234b67194f0d4ac8d3961e9040f217ec9768e27451c279105aa3391090`; library `ad57578dc2ed4fe3707da74a4f86620e758b1aa30064016785d187a5e86d3207`; starting checkpoint `9931750c5015ae5a385ac42202485868b0338530918421234c5aacbfe9b3e3b5`; exact reference bytes, ABI, provenance, E1 actor, and value initialization unchanged |
| Training | Design SHA-256 `84543f08265dae5076697549f25ce69b7e5e87947d4bb6e32b86a7700d24e67e`; PPO seeds `121001,121101,121201,121301,121401`; `1,048,576` transitions per seed; four environments, two expert-hold composition and two expert-reference rehearsal; first eight rollouts restricted, then full actor; final checkpoint only; no retries |
| Pairing | `study_pairing_sha256=4af9c9539bb2c7f9720f3e98a3f195c2b297d6bfe6adc51a1d63e7a8e4ad2464`; action, minibatch, RSI order/class/start, and evaluation streams derive from it in the uncalled adapter. Adapter receipt `01c591c566554c386fbfa38cdfc1019579e20af4a3c9a3e72dca958b09fed1f1`; integrated runtime pairing receipt **TBD pending Astra acceptance** |
| Evaluation | Seeds `97001`-`97020`; deterministic actions; expert start; full 1,000-step sufficient traces; direct-state evaluator/report design SHA-256 `d8f54f80051e4e3e58a540fc1414e9a971c9e299d903c2a00162bfebaa64a04b`; report schema `t2_reward_study_report/v1`; calibration `none` |
| Primary endpoint | Per checkpoint, mean over 20 episode values; each episode value is the mean of all `1,000` absolute per-step COM speed errors against `3.0 m/s`; no censoring |
| Arm estimator | Five paired PPO-seed differences `d_s = Y_baseline - Y_candidate`; report every pair, mean, median, range, paired 95% t interval using the five policies, and the exact one-sided sign result; pooled `n = 100` inference is forbidden |
| Task-improvement rule | All five `d_s > 0` and mean reduction `>= 0.25 m/s` |
| Safety | Candidate: `0/100` episodes with a fall, forbidden contact, invalid action, or incomplete trace; baseline viability is required first |
| Tracking | Each episode must keep all six whole-episode RMSE components within the frozen scales; checkpoint `>=16/20`; arm `>=4/5`; a candidate failure after a baseline pass breaks tracking |
| Overall candidate pass | Task rule met and baseline viable and candidate safety passed and candidate tracking passed |
| Tradeoff | Task, safety, tracking, and all six tracking components remain separate; no weighted aggregate |
| Secondary endpoints | Six tracking RMSEs; protected alpha/beta-independent task return `sum_t [1 - min(1, abs(v_t - 3) / 3)]`; fraction of steps in `[2.75,3.25] m/s`; speed quantiles and longest out-of-band run; fall-only first-fall step; fall and contact counts; descriptive stock return. Candidate `sum r_task` and `sum r_train` are diagnostics only. |
| Evidence | Both cohorts, if authorized and run, are `exploratory_fine_tuning_cycle`; claim ceiling: the candidate met or did not meet the preregistered T2 criterion in this matched five-seed implementation |
| Cycles | Pre-cycle freezes artifacts, runs the one F3 call, admits the candidate, and seals both arms. Cycle 0 is baseline; cycle 1 is candidate; cycle 2 prepares a protected-feedback revision under a separately reviewed protocol. Each cohort requires a new reservation and Samuel authorization. |
| Risk mitigations | Per-step absolute error; band occupancy and speed quantiles; separate fall, contact, posture, and tracking gates; static and observed reward-stream diagnostics; beta reported separately; expert rehearsal; staged unfreeze; final checkpoint only; evaluator excluded from the initial LLM prompt; same clean commit and host fingerprint; sequential arms |

## Pairing integration boundary

The launch note records no Astra acceptance message ID. Therefore
`src/oracle_composition/phase_b/training.py` is byte-untouched. The new adapter
refuses an arm-specific manifest hash when a study declares pairing and retains
the existing manifest-hash derivation only for undeclared single-arm runs. This
is an interface seam, not evidence that production actions, minibatches, RSI
assignments, resets, or evaluation episodes are paired.

## Execution gates and claim ceiling

- Do not run F3 from this slice.
- Do not run a training smoke or cohort until the candidate is admitted and the
  shared runtime consumes the reviewed pairing key with an integrated receipt.
- Every cohort still requires its own mailbox reservation; every full cohort
  requires Samuel's explicit authorization.
- No artifact or interface test in this package is behavioral evidence.

Claim ceiling:
`artifacts_and_interface_checks_only_no_reward_effect_tracking_utility_or_humanoid_competence_claim`.
