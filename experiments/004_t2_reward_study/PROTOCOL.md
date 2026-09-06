# Experiment 004: T2 expert-hold reward study

| status | current truth |
|---|---|
| progress | T2AR1 repairs the interface with direct-state formulas; reward-telemetry separation proven by test, reachable action-bound failures, verified-chain reference rows, exact trace-index replay, registry-resolved reward slots, and a content-addressed execution manifest. No candidate call, training, smoke, cohort, or behavioral evaluation ran. |
| bottleneck | Execution remains **NO-GO**: `t2_seal_v1.json` records `pairing_receipt: pending`; the candidate reward SHA-256 and an integrated `phase_b/training.py` pairing receipt are TBD. SCI-T2A-04 remains in `T2PAIR`. |
| next step | Fable reviews and commits T2AR1, then obtains Astra's accepted `T2PAIR` receipt and a separate dispatch verdict. Smoke and cohorts remain separately gated. |

## Frozen artifact ledger

| artifact | bytes | SHA-256 | evidence |
|---|---:|---|---|
| `oracle_expert_hold_v1.json` | 964 | `489b82591cf65034d58d30f921442a84c93c9c98885cb2f5b22dcaf645a77010` | canonical Phase B oracle; no transitions or recovery |
| `training_design_t2_v1.json` | 1,815 | `84543f08265dae5076697549f25ce69b7e5e87947d4bb6e32b86a7700d24e67e` | exact T2 Phase B design contract |
| `execution_manifest_t2_v1.json` | 3,428 | `d675b1ac02ad8713d995bd795ce2130acf41bc7a6fcc0a164167b24e3684ab3e` | frozen MDP/model/ABI/source/lock/commit/host seal; identical in both arm slots |
| `evaluator_design_t2_v1.json` | 1,808 | `7ae812d43c524285941cd567ce1663ff023cb6307229d9472a6dfe6577ebf5b3` | evaluator/report/protected-core source-bound design |
| `t2_reward_study_expert_hold_v1.json` | 5,872 | `4eb3440b943355b8eee96e7663a4d542833464a8720d4b8ac102c050d9623627` | common fields duplicated and equality-checked across arm slots |
| `t2_seal_v1.json` | 1,613 | `c0edc94a71d7e7cd23723e0e58b352568c2610d064db170a47a83869f02394ae` | non-model-facing hash ledger including the execution manifest; pairing receipt explicitly `pending`, dispatch withheld |
| `artifacts/experiments_004/t2_pairing_adapter_receipt_v1.json` | 1,736 | `6bd6f5ab3eb33ea563fff06c01828781d4c01864b7f0384108bfa5161c02540a` | fake-runtime adapter receipt; ignored local artifact, not trainer execution |
| `reward_study/pairing.py` | source | `724cda361d88e0ab7bec2de95bdf9c7811b156f410620af38a23821a53d15543` | arm-invariant adapter source bound into both study arms and its receipt |
| `reward_study/t2_evaluator.py` | source | `6f42b49b628ea11a2df13cf37a2ad170cde2be231427112103225e58b6f2e975` | protected direct-state computation and verified reference chain |
| `reward_study/t2_report.py` | source | `a534d9c745df4adf23e8e1d800d9f106aa4f55b21143c9a696458cc65afbbbaa` | input resolution, raw-trace replay, deterministic receipt, and separate telemetry writers |
| `phase_b/protected_metrics.py` | source | `a7456f038c08d6592a7bb4a92b4970e171ed38e6ac736cd477dcaa9f60760da6` | bound metric core preserving finite raw actions for independent bounds scoring |
| `phase_b/report_v2.py` | source | `a3a173c0717a0af8e1693c30e9ffef9e47a3fe876bdbde9132d5d6b2e1a324f2` | bound deterministic-receipt and telemetry schema authority |

The common arm-invariant pairing key is
`fd91156a949a4484b497a112327db864c2a4cbcbaf0cf1cf5db75064f6a5b3e0`.
It excludes only reward, arm label, output path, and timestamps. Changing the
common target from `3.0` to `3.1 m/s` changes the key to
`638ed9abbd81bd8e0453ceb4763c0b5604117f08489ef139f3c4524c2edf9432`
and changes the derived fake-runtime stream receipt.

### Astra integration re-seal

The accepted settling-band scoring repair changed `phase_b/report_v2.py` when
`3406bb5` was merged as `ed9f1d3`. The source-bound T2 chain was re-sealed before
the T2AR1 repairs: evaluator design `d8f54f80…` → `7339b06f…`, study manifest
`7aefaafc…` → `55b0b6e6…`, pairing key `4af9c953…` → `fb2fff65…`, pairing
adapter receipt `01c591c5…` → `a3b5baa9…`, and F3 seal `8c04f0c2…` →
`38dcab8d…`. No call, smoke, training, or evaluation run consumed the
superseded identities.

### T2AR1 execution and evaluator re-seal

The in-scope T2AR1 repairs added the execution manifest and changed the
source-bound evaluator, protected-metric, report, and pairing contracts. The
post-Astra identities were superseded as follows:

| identity | post-Astra seal | T2AR1 seal |
|---|---|---|
| execution manifest | absent | `d675b1ac02ad8713d995bd795ce2130acf41bc7a6fcc0a164167b24e3684ab3e` |
| evaluator design | `7339b06f271fdb053ae0c53d287b92c8b6cbd05fb67f18a33a8ab01cdfb4dbd7` | `7ae812d43c524285941cd567ce1663ff023cb6307229d9472a6dfe6577ebf5b3` |
| study manifest | `55b0b6e60216806ff873b21010a8a8eacd1c27570da24dace611fa1c389e1cd4` | `4eb3440b943355b8eee96e7663a4d542833464a8720d4b8ac102c050d9623627` |
| study pairing key | `fb2fff65ef5cdbc95a52d88c90a4ec9f22d62d1ba7092abd37b0bff956adb053` | `fd91156a949a4484b497a112327db864c2a4cbcbaf0cf1cf5db75064f6a5b3e0` |
| adapter receipt | `a3b5baa94fbdb818e02d77f84c775a5a2c647754d65b9570d166401ab377d27d` | `6bd6f5ab3eb33ea563fff06c01828781d4c01864b7f0384108bfa5161c02540a` |
| F3 seal | `38dcab8d4f6b52c47c1981a4b2347f1c15d66a1f2ff17729832a25678bbda448` | `c0edc94a71d7e7cd23723e0e58b352568c2610d064db170a47a83869f02394ae` |

No candidate call, smoke, training, or evaluation run used either superseded
seal chain.

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
| Oracle | `expert_hold/v1`; SHA-256 `489b82591cf65034d58d30f921442a84c93c9c98885cb2f5b22dcaf645a77010`; hold expert from verified-chain boundary `0`; no transitions or recovery; expert-reference RSI only. A `medium_hold/v1` route would be a new preregistered study. |
| Arms | `tracking_only/v1` SHA-256 `eea2b6a9893e6e4ca5aea5d6787580f12062084e758cc9c27db2f1376bcb1c5f` versus one admitted `target_speed_triangular_affine_t2_adapter/v1` specification SHA-256 **TBD after the single F3 call** |
| Candidate timing | The independent evaluator is frozen first; one initial F3 call; admit the candidate and seal both arm manifests before exposing baseline measurements |
| Common references and initialization | Corpus `a3e9a7234b67194f0d4ac8d3961e9040f217ec9768e27451c279105aa3391090`; library `ad57578dc2ed4fe3707da74a4f86620e758b1aa30064016785d187a5e86d3207`; starting checkpoint `9931750c5015ae5a385ac42202485868b0338530918421234c5aacbfe9b3e3b5`; exact reference bytes, ABI, provenance, E1 actor, and value initialization unchanged |
| Training | Design SHA-256 `84543f08265dae5076697549f25ce69b7e5e87947d4bb6e32b86a7700d24e67e`; PPO seeds `121001,121101,121201,121301,121401`; `1,048,576` transitions per seed; four environments, two expert-hold composition and two expert-reference rehearsal; first eight rollouts restricted, then full actor; final checkpoint only; no retries |
| Pairing | `study_pairing_sha256=fd91156a949a4484b497a112327db864c2a4cbcbaf0cf1cf5db75064f6a5b3e0`; action, minibatch, RSI order/class/start, and evaluation streams derive from it in the uncalled adapter. Adapter receipt `6bd6f5ab3eb33ea563fff06c01828781d4c01864b7f0384108bfa5161c02540a`; integrated runtime pairing receipt **TBD pending Astra acceptance** |
| Evaluation | Seeds `97001`-`97020`; deterministic actions; expert start; full 1,000-step sufficient traces; direct-state formulas; reward-telemetry separation proven by test; verified library/corpus/index/bundle/payload/row chain; evaluator/report design SHA-256 `7ae812d43c524285941cd567ce1663ff023cb6307229d9472a6dfe6577ebf5b3`; report schema `t2_reward_study_report/v1`; calibration `none` |
| Primary endpoint | Per checkpoint, mean over 20 episode values; each episode value is the mean of all `1,000` absolute per-step COM speed errors against `3.0 m/s`; no censoring |
| Arm estimator | Five paired PPO-seed differences `d_s = Y_baseline - Y_candidate`; report every pair, mean, median, range, paired 95% t interval using the five policies, and the exact one-sided sign result; pooled `n = 100` inference is forbidden |
| Task-improvement rule | All five `d_s > 0` and mean reduction `>= 0.25 m/s` |
| Safety | Candidate: `0/100` episodes with a fall, forbidden contact, invalid action, or incomplete trace; baseline viability is required first |
| Tracking | Each episode must keep all six whole-episode RMSE components within the frozen scales; checkpoint `>=16/20`; arm `>=4/5`; a candidate failure after a baseline pass breaks tracking |
| Overall candidate pass | Task rule met and baseline viable and candidate safety passed and candidate tracking passed |
| Tradeoff | Task, safety, tracking, and all six tracking components remain separate; no weighted aggregate |
| Secondary endpoints | Six tracking RMSEs; protected task return `sum_t [1 - min(1, abs(v_t - 3) / 3)]`; fraction of steps in `[2.75,3.25] m/s`; speed quantiles and longest out-of-band run; fall-only first-fall step; fall, contact, and invalid-action counts. Stock and candidate reward outputs are diagnostics only in `reward_diagnostics_t2_v1.json`, keyed by the scientific-receipt SHA-256. |
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
