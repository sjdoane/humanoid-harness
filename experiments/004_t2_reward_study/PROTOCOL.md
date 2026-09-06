# Experiment 004: T2 expert-hold reward study

| status | current truth |
|---|---|
| progress | T2C1 publishes the recipe-derived candidate, registry-resolves its exact canonical bytes, and regenerates the execution manifest, both arm bindings, study manifest, pairing key, and T2 seal as one `final_ready` set at verified clean HEAD `28067291bf43bb315e1702fc66c649310d4a3c97`. |
| bottleneck | Cycle 0 remains **NO-GO**: no baseline measurement, smoke, training, cohort, protected evaluation, reward effect, or behavioral evidence exists; resource reservation, Samuel's authorization, and review of this admission slice remain pending. |
| next step | Fable reviews and commits the complete T2C1 slice, then proposes the separately gated cycle-0 reservation to Astra and obtains Samuel's explicit authorization before any runtime command. |

## Frozen artifact ledger

| artifact | bytes | SHA-256 | evidence |
|---|---:|---|---|
| `oracle_expert_hold_v1.json` | 964 | `489b82591cf65034d58d30f921442a84c93c9c98885cb2f5b22dcaf645a77010` | canonical Phase B oracle; no transitions or recovery |
| `training_design_t2_v1.json` | 1,815 | `84543f08265dae5076697549f25ce69b7e5e87947d4bb6e32b86a7700d24e67e` | exact T2 Phase B design contract |
| `candidate_target_speed_t2_v1.json` | 2,105 | `c09e93dc27f129f0f65ed5be515b114a77673fef95027422f60ddade76dcc123` | canonical `target_speed_triangular_affine_t2_adapter/v1` specification; parameters reproduced from the exact retained recipe and registry-resolved |
| `execution_manifest_t2_v1.json` | 3,585 | `1ce2a4751f4e4149012a1cb0bbc88ee5a2fdaa497b616653d11a6146e0bd5eba` | final-ready frozen MDP/model/ABI/source/lock/host seal; clean execution HEAD verified as `28067291bf43bb315e1702fc66c649310d4a3c97` |
| `evaluator_design_t2_v1.json` | 1,808 | `634ea93e975e5331f9c71c5123a75ca525cc366055312bf4b3a4652dba772708` | evaluator/report/protected-core source-bound design |
| `t2_reward_study_expert_hold_v1.json` | 5,999 | `2fea955d94719e455920fe65eaf017e746313353fd2246a69190ea4db623f6e5` | `final_ready`; exact baseline/candidate bindings and final execution binding; integrated receipt unchanged |
| `t2_seal_v1.json` | 1,684 | `89aff467d05414553439ac5cfed6b5679d94ae08e455d1b687e3b9d685190624` | final admission ledger; state `candidate_admitted_no_further_initial_dispatch`; cannot authorize another F3 call |
| `f3_call/ledger_v1.json` | 1,636 | `cacfab1d4f3ad60881a315a388952ab362a4aacef158c18977e599bc61e1221b` | binds the exact proposal, recipe, model-call receipt, derived candidate, and `F3_INITIAL_CALL_RESULT.md`; no raw run bytes committed |
| `artifacts/experiments_004/t2_pairing_adapter_receipt_v1.json` | 1,736 | `6bd6f5ab3eb33ea563fff06c01828781d4c01864b7f0384108bfa5161c02540a` | superseded adapter receipt; report admission refuses it when labeled integrated |
| `pairing_receipt_v1.json` | 9,770 | `1a2b7ece139974117fd5c75e040d9cc52cd51a4e42c9b5afd794a60b02232348` | integrated shared-runtime interface receipt; immutable predecessor study lineage `8e81792a…2d10`; five seeds x four slots |
| `reward_study/pairing.py` | source | `8bdd0e3b0f2ba0b4c867a1d2be12869c59e7e59d1a584b22f9d093f19490f627` | exact-byte pairing authority bound by the current study and receipt |
| `reward_study/t2_evaluator.py` | source | `6f42b49b628ea11a2df13cf37a2ad170cde2be231427112103225e58b6f2e975` | protected direct-state computation and verified reference chain |
| `reward_study/t2_report.py` | source | `50bce64dce17151e5ef94494e27963fc58e9fc31900b9275c071807bea11e4f8` | final-ready reconciliation, input resolution, raw-trace replay, deterministic receipt, and separate telemetry writers |
| `phase_b/protected_metrics.py` | source | `a7456f038c08d6592a7bb4a92b4970e171ed38e6ac736cd477dcaa9f60760da6` | bound metric core preserving finite raw actions for independent bounds scoring |
| `phase_b/report_v2.py` | source | `a3a173c0717a0af8e1693c30e9ffef9e47a3fe876bdbde9132d5d6b2e1a324f2` | bound deterministic-receipt and telemetry schema authority |

The common arm-invariant pairing key is
`ccef84c7f05992556183c9ebdacb86945fe3e7832cd1e90a03eb9544096309a2`.
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

### T2PAIRR1 pairing-review repair re-seal

PAIR-01 changed the exact `phase_b/contracts.py` bytes from
`c8b6fd3cb61f0a79a086156f1a3440e64a5dae7395086735c59a886b2ec8448a`
to `c25d61de985f06ab69945842185fd004be3a623d3c984df11718ca1af53855f1`.
PAIR-02 also changed the source-bound training and runtime bytes. The receipt
generator was repaired to create its controlled candidate in the required
`final_ready` validation state, changing `reward_study/pairing.py` from
`6fe7aae67bd23557eb5e914e366bace5fcec7273c271b5c242d6f2546ce7261f`
to `8bdd0e3b0f2ba0b4c867a1d2be12869c59e7e59d1a584b22f9d093f19490f627`.

| identity | T2AR2 seal | T2PAIRR1 seal |
|---|---|---|
| pairing receipt | `e5351c3b49ba97cc362ccd68a0bcf797c5077fb0c2ace78535b24e5567e0a259` | `1a2b7ece139974117fd5c75e040d9cc52cd51a4e42c9b5afd794a60b02232348` |
| execution manifest | `9492cc639cf9eb83f98098944e136b5a9a5e4f581dfeff3034bf2a58b965e368` | `7e303034758203612fb62bf5c8193bf93731a5b322303c4a53dacd13f56d8d1c` |
| evaluator design | `634ea93e975e5331f9c71c5123a75ca525cc366055312bf4b3a4652dba772708` | `634ea93e975e5331f9c71c5123a75ca525cc366055312bf4b3a4652dba772708` |
| study manifest | `8e81792ab6b4847776ce4cd352bb4eda6c0f705ceaf9ff644a908508c8982d10` | `a839362aad12392612e66cc1c6479e904e5c037e77a37d96d6e9025f2619eecb` |
| study pairing key | `64529d781ae3fb5030ce6d018504c69e31e6e62307775c8e47cdb8c81996c1e7` | `affe8347f934bcbb5d19ec96cdd71f7bf38f32a5a0f9f83446cb8b661b11ec19` |
| F3 seal | `924769fe26bae38e84e244fcbeea89f24b72b2ec881f12a3261ec842f7ea059e` | `6ce006bb983611179ab9fc8bc47b7b01c74d6303a3f7f0a917bd35947d9d5a9d` |

The pairing receipt remains 9,770 bytes, but its bytes **did change**. It binds
the repaired contract, training, runtime, and pairing source bytes and retains
the T2AR2 study hash and pairing key as immutable predecessor lineage. The
evaluator design was regenerated and remained byte-identical. No run consumed
the superseded T2AR2 chain.

### T2C1 candidate admission and final-ready re-seal

At clean HEAD `28067291bf43bb315e1702fc66c649310d4a3c97`, T2C1 parsed the
exact 73-byte retained recipe
`078fabc83ff3d572789b268c05ccff5a08050e00d0fd104399e2294d3b91ec28`.
The recipe's parameters—not manually entered values—constructed a
`TargetSpeedRewardSpec`; its canonical bytes then reloaded through
`RewardRegistry` without change. Admission regenerated the execution manifest,
both arm bindings, study manifest, pairing key, and T2 seal as one set.

| identity | T2PAIRR1 pending seal | T2C1 final-ready seal |
|---|---|---|
| candidate reward | `TBD` | `c09e93dc27f129f0f65ed5be515b114a77673fef95027422f60ddade76dcc123` |
| execution commit | pending final admission | `28067291bf43bb315e1702fc66c649310d4a3c97` verified as clean HEAD |
| execution manifest | `7e303034758203612fb62bf5c8193bf93731a5b322303c4a53dacd13f56d8d1c` | `1ce2a4751f4e4149012a1cb0bbc88ee5a2fdaa497b616653d11a6146e0bd5eba` |
| study manifest | `a839362aad12392612e66cc1c6479e904e5c037e77a37d96d6e9025f2619eecb` | `2fea955d94719e455920fe65eaf017e746313353fd2246a69190ea4db623f6e5` |
| study pairing key | `affe8347f934bcbb5d19ec96cdd71f7bf38f32a5a0f9f83446cb8b661b11ec19` | `ccef84c7f05992556183c9ebdacb86945fe3e7832cd1e90a03eb9544096309a2` |
| T2 seal | `6ce006bb983611179ab9fc8bc47b7b01c74d6303a3f7f0a917bd35947d9d5a9d` | `89aff467d05414553439ac5cfed6b5679d94ae08e455d1b687e3b9d685190624` |

The exact superseded T2PAIRR1 study and seal bytes are retained at
`experiments/004_t2_reward_study/superseded/superseded_t2pairr1_study_manifest_v1.json`
and
`experiments/004_t2_reward_study/superseded/superseded_t2pairr1_seal_v1.json`.
Report regression
accepts a well-formed synthetic report bound to the published final-ready set
and still rejects the same report when rebound to the superseded study with
`study_not_final_ready`. This is admission evidence only, not a baseline or
candidate measurement.

The committed `f3_call/` ledger contains only the exact proposal
(`060762c7…e07b`), trusted recipe (`078fabc8…ec28`), and model-call receipt
(`1a8d0e78…a40c`) JSON plus a binding ledger. It commits none of the retained
raw run bytes and binds the call records and derived candidate to the 4,793-byte
`F3_INITIAL_CALL_RESULT.md` at
`07cbe7eb7a41ca9372226557eecd003ab31c225a3b82abca2055ca8b4aa0f34f`.

## Frozen protocol `t2_reward_study_expert_hold/v1`

| field | frozen value |
|---|---|
| Family and authorable factor | Family B; the reward specification only |
| Task | Stock `Humanoid-v5`; expert start; target COM forward speed `3.0 m/s`; `1,000` steps |
| Oracle | `expert_hold/v1`; SHA-256 `489b82591cf65034d58d30f921442a84c93c9c98885cb2f5b22dcaf645a77010`; hold expert from verified-chain boundary `0`; no transitions or recovery; expert-reference RSI only. A `medium_hold/v1` route would be a new preregistered study. |
| Arms | `tracking_only/v1` SHA-256 `eea2b6a9893e6e4ca5aea5d6787580f12062084e758cc9c27db2f1376bcb1c5f` versus admitted `target_speed_triangular_affine_t2_adapter/v1` SHA-256 `c09e93dc27f129f0f65ed5be515b114a77673fef95027422f60ddade76dcc123` |
| Candidate timing | The independent evaluator was frozen first; the one initial F3 call was retained; T2C1 admitted its recipe-derived candidate and sealed both arms before any baseline measurement |
| Common references and initialization | Corpus `a3e9a7234b67194f0d4ac8d3961e9040f217ec9768e27451c279105aa3391090`; library `ad57578dc2ed4fe3707da74a4f86620e758b1aa30064016785d187a5e86d3207`; starting checkpoint `9931750c5015ae5a385ac42202485868b0338530918421234c5aacbfe9b3e3b5`; exact reference bytes, ABI, provenance, E1 actor, and value initialization unchanged |
| Training | Design SHA-256 `84543f08265dae5076697549f25ce69b7e5e87947d4bb6e32b86a7700d24e67e`; PPO seeds `121001,121101,121201,121301,121401`; `1,048,576` transitions per seed; four environments, two expert-hold composition and two expert-reference rehearsal; first eight rollouts restricted, then full actor; final checkpoint only; no retries |
| Pairing | `t2_reward_pairing/v1`; final-ready `study_pairing_sha256=ccef84c7f05992556183c9ebdacb86945fe3e7832cd1e90a03eb9544096309a2`. The runtime accepts it only through exact canonical study bytes plus both exact embedded arm manifests. Action noise is indexed by rollout and environment slot; minibatch permutations by update; composition blocks and reset/RSI block, origin, class, and start by global episode index within each slot; declared evaluation seeds are the actual reset seeds at declared indices. The integrated receipt remains exactly `1a2b7ece139974117fd5c75e040d9cc52cd51a4e42c9b5afd794a60b02232348`. |
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
| Cycles | Pre-cycle artifacts, the one F3 call, candidate admission, and both arm seals are complete. Cycle 0 is baseline; cycle 1 is candidate; cycle 2 prepares a protected-feedback revision under a separately reviewed protocol. Each cohort requires a new reservation and Samuel authorization. |
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

The PAIR-02 regression additionally sends both reward arms through
`run_ppo_training` and the controlled fake runtime, observes the action-noise
arrays and minibatch permutations at their production consumers, and compares
each environment slot's reset ledger plus the RSI ledger. A test seam disables
the shared paired route; the same regression then detects the misroute. This is
production-routing evidence with fake environments, not a simulator or behavior
result.

## Execution gates and claim ceiling

- The one initial F3 call is exhausted; the final seal explicitly refuses any
  further initial dispatch.
- Do not expose or compute a baseline measurement, run a smoke, or start a
  cohort until this admission slice is accepted, its exact committed identities
  are read back, a resource reservation is accepted, and Samuel authorizes the
  specific runtime action.
- Every cohort still requires its own mailbox reservation; every full cohort
  requires Samuel's explicit authorization.
- No artifact or interface test in this package is behavioral evidence.

Claim ceiling:
`artifacts_and_interface_checks_only_no_reward_effect_tracking_utility_or_humanoid_competence_claim`.
