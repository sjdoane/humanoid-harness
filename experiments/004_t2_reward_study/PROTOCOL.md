# Experiment 004: T2 expert-hold reward study

| status | current truth |
|---|---|
| progress | T2AR2 binds the reviewed `pairing_receipt_v1.json`, re-seals the evaluator, pending execution manifest, matched-arm study, and F3 ledger, and adds a fail-closed `final_ready` report-admission gate. Report hashes named files and replays traces; cross-manifest reconciliation is enforced by the final-ready gate. No candidate call, smoke, training, cohort, or behavioral evaluation ran. |
| bottleneck | Execution remains **NO-GO**: baseline is registry-resolved; the candidate resolver is unit-tested and the canonical candidate is TBD. Report admission therefore stops with `study_not_final_ready`; the actual execution commit and clean tree cannot be sealed until candidate admission. |
| next step | Fable independently reviews and commits T2AR2, then obtains a separate dispatch verdict. If a candidate is later accepted, the admission step must registry-resolve both rewards, verify the actual clean execution HEAD, and regenerate the execution manifest, study manifest, and seal together before any report or runtime decision. |

## Frozen artifact ledger

| artifact | bytes | SHA-256 | evidence |
|---|---:|---|---|
| `oracle_expert_hold_v1.json` | 964 | `489b82591cf65034d58d30f921442a84c93c9c98885cb2f5b22dcaf645a77010` | canonical Phase B oracle; no transitions or recovery |
| `training_design_t2_v1.json` | 1,815 | `84543f08265dae5076697549f25ce69b7e5e87947d4bb6e32b86a7700d24e67e` | exact T2 Phase B design contract |
| `execution_manifest_t2_v1.json` | 3,554 | `9492cc639cf9eb83f98098944e136b5a9a5e4f581dfeff3034bf2a58b965e368` | frozen MDP/model/ABI/source/lock/host seal; records launch-base provenance; the execution commit is verified at admission |
| `evaluator_design_t2_v1.json` | 1,808 | `634ea93e975e5331f9c71c5123a75ca525cc366055312bf4b3a4652dba772708` | evaluator/report/protected-core source-bound design |
| `t2_reward_study_expert_hold_v1.json` | 5,908 | `8e81792ab6b4847776ce4cd352bb4eda6c0f705ceaf9ff644a908508c8982d10` | pending state; exact common bindings; integrated receipt bound; candidate remains `TBD` |
| `t2_seal_v1.json` | 1,671 | `924769fe26bae38e84e244fcbeea89f24b72b2ec881f12a3261ec842f7ea059e` | non-model-facing ledger; `pairing_receipt` is the integrated receipt SHA-256; dispatch awaits a separate verdict |
| `artifacts/experiments_004/t2_pairing_adapter_receipt_v1.json` | 1,736 | `6bd6f5ab3eb33ea563fff06c01828781d4c01864b7f0384108bfa5161c02540a` | superseded adapter receipt; report admission refuses it when labeled integrated |
| `pairing_receipt_v1.json` | 9,770 | `e5351c3b49ba97cc362ccd68a0bcf797c5077fb0c2ace78535b24e5567e0a259` | integrated shared-runtime interface receipt; immutable predecessor study lineage `4eb3440b…3627`; five seeds x four slots |
| `reward_study/pairing.py` | source | `6fe7aae67bd23557eb5e914e366bace5fcec7273c271b5c242d6f2546ce7261f` | exact-byte pairing authority bound by the current study and receipt |
| `reward_study/t2_evaluator.py` | source | `6f42b49b628ea11a2df13cf37a2ad170cde2be231427112103225e58b6f2e975` | protected direct-state computation and verified reference chain |
| `reward_study/t2_report.py` | source | `50bce64dce17151e5ef94494e27963fc58e9fc31900b9275c071807bea11e4f8` | final-ready reconciliation, input resolution, raw-trace replay, deterministic receipt, and separate telemetry writers |
| `phase_b/protected_metrics.py` | source | `a7456f038c08d6592a7bb4a92b4970e171ed38e6ac736cd477dcaa9f60760da6` | bound metric core preserving finite raw actions for independent bounds scoring |
| `phase_b/report_v2.py` | source | `a3a173c0717a0af8e1693c30e9ffef9e47a3fe876bdbde9132d5d6b2e1a324f2` | bound deterministic-receipt and telemetry schema authority |

The common arm-invariant pairing key is
`64529d781ae3fb5030ce6d018504c69e31e6e62307775c8e47cdb8c81996c1e7`.
It excludes only reward, arm label, output path, and timestamps. Any common
field change changes the key and creates a different execution family.

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

### T2AR2 final-ready and T2PAIR re-seal

T2AR2 binds the immutable T2PAIR receipt while preserving its explicit
predecessor lineage. It does not rewrite or regenerate that receipt. The
T2AR1 identities are superseded as follows:

| identity | T2AR1 seal | T2AR2 seal |
|---|---|---|
| execution manifest | `d675b1ac02ad8713d995bd795ce2130acf41bc7a6fcc0a164167b24e3684ab3e` | `9492cc639cf9eb83f98098944e136b5a9a5e4f581dfeff3034bf2a58b965e368` |
| evaluator design | `7ae812d43c524285941cd567ce1663ff023cb6307229d9472a6dfe6577ebf5b3` | `634ea93e975e5331f9c71c5123a75ca525cc366055312bf4b3a4652dba772708` |
| study manifest | `4eb3440b943355b8eee96e7663a4d542833464a8720d4b8ac102c050d9623627` | `8e81792ab6b4847776ce4cd352bb4eda6c0f705ceaf9ff644a908508c8982d10` |
| study pairing key | `fd91156a949a4484b497a112327db864c2a4cbcbaf0cf1cf5db75064f6a5b3e0` | `64529d781ae3fb5030ce6d018504c69e31e6e62307775c8e47cdb8c81996c1e7` |
| integrated pairing receipt | unbound | `e5351c3b49ba97cc362ccd68a0bcf797c5077fb0c2ace78535b24e5567e0a259` |
| F3 seal | `c0edc94a71d7e7cd23723e0e58b352568c2610d064db170a47a83869f02394ae` | `924769fe26bae38e84e244fcbeea89f24b72b2ec881f12a3261ec842f7ea059e` |

The receipt's `source_study_manifest_sha256` remains the T2AR1 study hash and
its recorded pairing key remains the T2PAIR key; the current study binds the
receipt by its exact immutable SHA-256 and records the post-pairing re-seal as a
new common-field lineage. No run consumed either lineage.

F3 preparation must verify the exact seal bytes plus every sealed file before
rendering the model prompt. The seal and its evaluator/study-manifest contents
are retained outside the 6,011-byte prompt. The receipt is now verified, but
`withheld_pending_dispatch_verdict` remains a verified preparation state, not
dispatch authority.

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
| Pairing | `t2_reward_pairing/v1`; current pending-study `study_pairing_sha256=64529d781ae3fb5030ce6d018504c69e31e6e62307775c8e47cdb8c81996c1e7`. The runtime accepts it only through exact canonical study bytes plus both exact embedded arm manifests. Action noise is indexed by rollout and environment slot; minibatch permutations by update; composition blocks and reset/RSI block, origin, class, and start by global episode index within each slot; declared evaluation seeds are the actual reset seeds at declared indices. The integrated receipt is exactly `e5351c3b49ba97cc362ccd68a0bcf797c5077fb0c2ace78535b24e5567e0a259`. |
| Evaluation | Seeds `97001`-`97020`; deterministic actions; expert start; full 1,000-step sufficient traces; direct-state formulas; reward-telemetry separation proven by test; verified library/corpus/index/bundle/payload/row chain; evaluator/report design SHA-256 `634ea93e975e5331f9c71c5123a75ca525cc366055312bf4b3a4652dba772708`; report schema `t2_reward_study_report/v1`; calibration `none` |
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
| Risk mitigations | Per-step absolute error; band occupancy and speed quantiles; separate fall, contact, posture, and tracking gates; static and observed reward-stream diagnostics; beta reported separately; expert rehearsal; staged unfreeze; final checkpoint only; evaluator excluded from the initial LLM prompt; actual clean execution commit verified at candidate admission; sequential arms |

## Pairing integration boundary

Astra accepted the exact executable scope in mailbox message
`20260906T072706.830333Z-a50cb508191d4d629b4ba6c8f9e42103`; Fable launched
this slice at clean base `7698256`. A paired `TrainingPlan` cannot be built from
a supplied digest: it requires canonical study bytes and byte-matching embedded
baseline and candidate arm manifests, recomputes the common projection, and
checks that the plan's ordinary execution identity is one of the two distinct
arm hashes before any policy or environment factory is called. Missing,
mismatched, stale, or forged keys fail there. Undeclared single-arm plans are
explicitly `non_paired` and preserve the original manifest-hash derivation.

The integrated fake-runtime receipt covers primitive variates and indexed
schedules, not actions, states, returns, episode lengths, or learned behavior.
Per-slot episode counters keep reset schedules aligned even if the two policies
later terminate at different times. Action-bound failure is interface-reachable;
no production T2 trace producer has run. The re-sealed receipt binding does not
authorize dispatch, smoke, or training.

## Execution gates and claim ceiling

- Do not run F3 from this slice.
- Do not run a training smoke or cohort until the candidate is admitted and the
  shared runtime consumes the reviewed pairing key with an integrated receipt.
- Every cohort still requires its own mailbox reservation; every full cohort
  requires Samuel's explicit authorization.
- No artifact or interface test in this package is behavioral evidence.

Claim ceiling:
`artifacts_and_interface_checks_only_no_reward_effect_tracking_utility_or_humanoid_competence_claim`.
