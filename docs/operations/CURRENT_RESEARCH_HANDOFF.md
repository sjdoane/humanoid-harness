# Current research handoff

| status | current truth |
|---|---|
| progress | `TASK-20260906-T2A` freezes the canonical expert-hold oracle, expert-only T2 training design, source-bound direct-state evaluator/report, matched-arm study manifest, and source-bound uncalled pairing adapter. The 25 focused tests pass; the release-byte full suite adds no failure beyond the recorded sandbox baseline. |
| bottleneck | T2 execution remains **NO-GO**. The candidate reward SHA-256 and Astra-accepted integrated trainer pairing receipt are both TBD; the local adapter receipt is an interface check and does not show that `phase_b/training.py` consumes the pairing key. No candidate call, smoke, training, cohort, or behavioral evaluation ran. |
| next step | Fable reviews and commits T2A (force-add the intentionally ignored pairing receipt), then—only if the separate F3PIN review accepts—runs the one initial F3 call and admits/seals the candidate. Astra must explicitly accept and integrate the shared trainer seam before any smoke or cohort proposal; every cohort still needs its own reservation and Samuel authorization. |

Fable integration note (2026-09-06T05:56Z): T2A is committed in this same commit with the force-added pairing adapter receipt. F3PIN review verdict: ACCEPT_F3_REPIN_STATIC_ONLY, WITHHOLD_DISPATCH; repairs in packet TASK-20260906-F3PINR1 (next writer). The pairing seam (packet TASK-20260906-T2PAIR) launches only after Astra accepts proposal 20260906T055257. A read-only review of T2A runs in parallel. Fable resume: read the F3PINR1 final and the T2A review; commit; if Astra accepted the pairing terms, launch T2PAIR; the single F3 call waits for the T2 seal, the pairing receipt, and a dispatch verdict; cohorts wait for a reservation and Samuel's authorization.

T2A completion (2026-09-06T05:50Z): base/head `077e4e8`; expert-hold oracle
`489b8259…7010`, T2 training design `84543f08…e67e`, evaluator design
`d8f54f80…a04b`, matched-arm manifest `7aefaafc…0738`, arm-invariant pairing
key `4af9c953…2464`, and adapter receipt `01c591c5…1f1` all reload from
canonical bytes and their source/artifact bindings. The unchanged T1 training
design remains `1d104a52…c69e`, and `src/oracle_composition/phase_b/training.py`
is byte-untouched. The evaluator recomputes COM speed and six tracking errors
from direct state and authoritative expert rows; the deterministic report keeps
reward diagnostics and host/wall telemetry outside protected endpoints, uses
five final checkpoints as the independent units, and reports incomplete traces
as safety failures without pooled inference.

Focused T2 validation is `25 passed` in `5.07 s`. Ruff lint and the authorized
`243`-file source/test format check pass, as does `git diff --check`. The final
full suite, explicitly deselecting
`tests/experiments/test_reward_target_speed_manifest.py::test_recorded_no_learning_runtime_receipt_replays_exactly`,
is `1782 passed, 18 skipped, 1 deselected, 44 failed` in `223.83 s`; the exact
set of all `44` failed node IDs equals
`artifacts/bootstrap_tqc_humanoid/sandbox_baseline_failures_03a3.txt`. An
earlier adjacent-only run had one transient fake-supervisor worker crash; its
isolated rerun and the release-byte full suite both passed, with no repair made.
The pairing adapter receipt lives at ignored path
`artifacts/experiments_004/t2_pairing_adapter_receipt_v1.json` and must be
force-added by Fable if retained. No Git write, F3/candidate call, standalone
simulator run, smoke, training, cohort, or protected behavioral evaluation ran.
Builder wall time was `64m` through release-byte full-suite reconciliation.

Fable resume: inspect the Experiment 004 protocol, canonical JSONs, protected
evaluator/report, study equality checks, and pairing negatives; force-add the
ignored adapter receipt and commit the complete slice. Read the independent
F3PIN verdict before dispatch. If accepted, perform exactly the authorized
single initial call, admit the candidate, update and seal both arms, then obtain
Astra's explicit pairing integration and a production-runtime receipt. Only
after those gates may Fable propose the disposable smoke or ask Samuel to
authorize cycle 0.

Fable integration note (2026-09-06T04:40Z): F3PIN committed as `b071fc5`. Reward lane: the T2 protocol is frozen in the strategy (`t2_reward_study_expert_hold/v1`, commit c0332b1); the pairing-key runtime change is proposed to Astra; builder `sol-builder-20260906-t2a` (packet `TASK-20260906-T2A`) builds the study artifacts; a targeted read-only review of F3PIN runs in parallel. Fable resume: read the T2A final and the F3PIN review; commit; if the review accepts, run the single F3 call under the re-pinned protocol (preparation and ingestion by Fable under its lease, candidate worker read-only), admit and seal the candidate, then propose the cycle-0 smoke and cohort reservation and ask Samuel for authorization 1.

F3PIN completion (2026-09-06T04:30Z): base `8d617e3`; expected checkout/branch/import origin are the Fable `main` checkout, preparation/ingestion owner is `fable-f3-prepare`, and the sole candidate owner is `fable-f3-initial`. The 1,773-byte baseline remains `eea2b6a9…1c5f`; FT2R3 did not change it. The deterministic 6,011-byte prompt is `4d1a2976…e2bbc` and states the expert-start T2 task, baseline/F2 identities, alpha/beta-only boundary, and absence of measured feedback without embedding protected payloads or an execution manifest. Focused F3/F1 validation is `117 passed`; Ruff lint, format, and `git diff --check` pass. The full suite excluding `tests/experiments/test_reward_target_speed_manifest.py::test_recorded_no_learning_runtime_receipt_replays_exactly` is `1757 passed, 18 skipped, 1 deselected, 44 failed` in `220.39 s`; the 44 failed node IDs exactly match `artifacts/bootstrap_tqc_humanoid/sandbox_baseline_failures_03a3.txt`. No candidate call, training, standalone simulator, or reward result occurred. See `docs/operations/dual-orchestration/F3_FABLE_REPIN.md`.

Fable resume: inspect the F3PIN diff, exact hashes, Fable expected configuration, deterministic dossier, and duplicate-intent refusal; commit the complete static slice; then freeze the matched T2 reward-study protocol. Do not dispatch the candidate call from this handoff.

Fable integration note (2026-09-06T04:11Z): FT2R3 committed as `c945379`; this closes Fable's tracker-lane work (ADR 0009 lane swap). The oracle and tracker lane, including the combined re-reviews of FT2R1 to FT2R3, the smoke reservation, and the next composition packet, is Astra's from this commit; see docs/operations/TRACKER_LANE_HANDOFF.md. Fable now owns the reward lane: re-pin the F3 one-call protocol (packet TASK-20260906-F3PIN), freeze the T2 reward-study protocol from survey sol-survey-20260906-t2proto, then reward cycle 0 and 1 once compute is authorized.

FT2R3 completion (2026-09-06T04:05Z): base/head `6d71e8b`; focused Phase B/harness/reward validation is `402 passed, 16 skipped` in `102.31 s`. The full suite is `1684 passed, 18 skipped, 1 deselected, 44 failed` in `218.30 s`; the sorted set of all `44` failed node IDs exactly matches `artifacts/bootstrap_tqc_humanoid/sandbox_baseline_failures_03a3.txt`. The sole deselection was the forbidden reward-lane receipt test `tests/experiments/test_reward_target_speed_manifest.py::test_recorded_no_learning_runtime_receipt_replays_exactly`; it was not executed or changed. Ruff lint, the authorized `235`-file source/test format check, `git diff --check`, Phase A byte identity/reproducibility, live E1 `68/68`, and documentation-to-artifact hash checks pass. No training or training smoke ran; simulator execution was limited to existing bounded tests.

Fable resume: inspect the FT2R3 diff and mechanism regressions; preserve the interface/isolation-only claim; commit the slice; launch one combined scientific and robustness re-review of FT2R1 through FT2R3; and do not run the smoke before both verdicts.

Fable integration note (2026-09-06T02:24Z): FT2R2 committed as `2930cd6`. Next writer: `sol-builder-20260906-ft2r3` (packet `TASK-20260906-FT2R3`, isolation and mechanism-level negatives; no training). Then one combined scientific and robustness re-review of FT2R1..FT2R3, then the smoke reservation proposal to Astra; the five-seed cohort needs Samuel's authorization. Fable resume: read the FT2R3 final; commit; launch the two re-reviews; do not run the smoke before the re-review verdicts.

FT2R2 completion (2026-09-06T02:16Z): base/head `87d2e39`; focused Phase B/harness/reward validation is `371 passed, 16 skipped` in `95.63 s`. The full suite is `1653 passed, 18 skipped, 1 deselected, 44 failed` in `273.63 s`; the sorted set of all `44` failed node IDs exactly matches `artifacts/bootstrap_tqc_humanoid/sandbox_baseline_failures_03a3.txt`. The sole deselection was the forbidden reward-lane receipt test `tests/experiments/test_reward_target_speed_manifest.py::test_recorded_no_learning_runtime_receipt_replays_exactly`; it was not executed or changed. Ruff lint, the authorized `229`-file source/test format check, `git diff --check`, Phase A byte identity/reproducibility, live E1 `68/68`, and documentation-to-artifact hash checks pass. Repository-wide `ruff format --check .` additionally inspects Markdown and reports only the pre-existing code fence in out-of-scope Astra document `docs/operations/dual-orchestration/TASK-F2-t2-evaluator.md`; `git diff` is empty for that file and blame attributes the flagged lines to `2c94710a`, so this lease did not modify them. No training or training smoke ran; simulator execution was limited to existing bounded no-learning tests.

The F2 entry binds formula `target_speed_triangular_affine_t2_adapter/v1` at `06439388…301`, parser `target_speed_triangular_affine_recipe/v1` at `c60dea03…dccb`, the exact 137-byte bounds at `2c603026…6933`, task-input source `9607f2d5…f0d2`, output `[-10,15]`, the stock-telemetry compositor, flat bounded `alpha`/`beta`, exploratory evidence class, and mandatory FT2R1 task-input admission consumption. Astra's evaluator and parser bytes were not changed. Handoff edits are committed by Fable's integration commit by design; this sentence is the FT2-SCI-09 documentation repair.

Fable resume: inspect the FT2R2 diff and validation, preserve the interface/integrity-only claim, commit the slice, launch FT2R3 for the explicitly deferred isolation findings, and run a combined re-review before any smoke. Do not treat F2 registration, green tests, or the fake runtime as behavioral evidence.

Fable integration note (2026-09-06T00:32Z): FT2R1 committed as `5965f0c`. Astra's F2 evaluator commit `9bbb658` is merged next; then writer `sol-builder-20260906-ft2r2` (packet `TASK-20260906-FT2R2`), then `FT2R3`, then one combined re-review, then the smoke reservation proposal to Astra and Samuel's cohort authorization. No training has run. Fable resume: read the FT2R2 final; commit; launch FT2R3; do not run the smoke before both repairs and the re-review.

FT2R1 completion (2026-09-06T00:24Z): base/head `a51ea9e`; focused harness/Phase B/reward validation is `287 passed, 16 skipped` in `99.10 s`. The final full suite is `1525 passed, 18 skipped, 1 deselected, 44 failed` in `250.15 s`; all `44` failed node IDs exactly match `artifacts/bootstrap_tqc_humanoid/sandbox_baseline_failures_03a3.txt`. The sole deselection was the forbidden reward-lane receipt test `tests/experiments/test_reward_target_speed_manifest.py::test_recorded_no_learning_runtime_receipt_replays_exactly`; it was not executed or changed. Ruff lint, the `332`-file format check, `git diff --check`, frozen Phase A byte identity/reproducibility, and documentation-to-artifact hash checks pass. No training or training smoke ran; simulator execution was limited to the existing bounded no-learning tests.

The FT1 `policy.py` bytes had to change to close ADV-03's pre-allocation NPZ defect; E1 was regenerated against live policy `fd1d8729…3fd1` and receipt-generator `9cc5914d…d0c0`. Final seals are E1 `5754db8e…34bc9`, starting checkpoint `9931750c…e3b5`, tracking-only reward `eea2b6a9…1c5f`, evaluator `4145c7ba…4f7d`, and reviewed training-admission manifest `45ce5baa…ebb`. The ignored step-0 actor export remains byte-identical at `6ebc2b56…cfe0`; Astra's reviewed `task_inputs_v2.py` remains unchanged at `9607f2d5…f0d2`, and no pending adapter was added to the sole-entry `tracking_only/v1` registry. Strict provenance regeneration produced cycle-1/2 provenance `690b0852…0838` / `b9a75768…c19c` and scientific receipts `c44ac70d…b8f2` / `2a8d1d77…0f0e`.

Fable resume: inspect the FT2R1 diff and the final full-suite baseline comparison; verify the E1/manifest/documentation seals and regenerated Phase A provenance chain; then commit this complete interface-only slice. Do not run training from this handoff. Collect and review the disjoint calibration receipt before task-success evaluation, obtain a mailbox reservation before the smoke, and obtain Samuel's explicit authorization before the cohort.

Fable integration note (2026-09-05T23:11Z): FT2 committed as `f6287e5`. Next writer: `sol-builder-20260905-ft2r1` (packet `TASK-20260905-FT2R1`, the twelve FT1 and E003R1 review findings; no training). Read-only reviews of FT2 run in parallel. Astra's F2 formula interface is accepted with the velocity-bound admission certificate condition; their reviewed adapter hash is pending. Fable resume: read the FT2R1 final and the FT2 reviews; commit; fold; then send Astra the smoke reservation proposal (seed 121901, 196,608 transitions, expected 3 min, hard 20 min) and ask Samuel for the five-seed cohort authorization before any cohort run.

FT2 completion (2026-09-05T23:04Z): the focused no-repeat-simulator set is `56 passed`; the full suite is `1492 passed, 18 skipped, 1 deselected, 44 failed` in `148.84 s`. All `44` failed node IDs exactly match `artifacts/bootstrap_tqc_humanoid/sandbox_baseline_failures_03a3.txt`. The sole deselection was `tests/experiments/test_reward_target_speed_manifest.py::test_recorded_no_learning_runtime_receipt_replays_exactly`; it was not executed or changed. The full run includes the bounded real FT1 200-step identity smoke and the certified rehearsal predecessor plus one counted step. The strict FT1 policy source remains byte-identical, and the worker E1 audit still binds all `68` fixtures and action SHA-256 `5a70c792…ceb568`.

Fable resume: review the FT2 diff and validation, commit the complete slice without changing its interface-only claim, then obtain the separate mailbox reservation before the disposable smoke and Samuel authorization before the five-seed cohort.

FT2 30-minute checkpoint (2026-09-05T22:32Z): lease remains `CLAIMED` by `sol-builder-20260905-ft2` with the exact packet scope. The sealed FT1 `policy.py` was restored byte-for-byte after a regression correctly rejected an E1 source-hash change; trained reload now lives only in FT2 persistence. Fable resume: read the final FT2 validation counts and diff, preserve the no-training claim ceiling, commit the slice, then request a separate mailbox reservation before any disposable smoke and Samuel authorization before a five-seed cohort.

Fable integration note (2026-09-05T21:43Z): E003R1 committed as `6dc946b` on top of the Astra integration merge `8d91a81` (M1 accepted merge plus the T2 input contract). Next writer: `sol-builder-20260905-ft2` (packet `TASK-20260905-FT2`, training worker, supervision, persistence, report v2 writer, CLI train, fake-runtime end-to-end; no real training). Read-only reviews of FT1 (scientific) and E003R1 (robustness) run in parallel. Fable resume: read the FT2 final and both reviews; commit FT2; fold review findings; then propose the disposable 20-minute smoke to Astra by mailbox reservation and ask Samuel for cohort authorization before any five-seed run.

E003R1 completion (2026-09-05T21:38Z): base `8d91a81`; exact execution-manifest SHA-256 `cccaf040…c00`; Phase A v1 report hashes remain `3d32dd…623`, `480d5b…5f0`, and `e677b9…94c`, and all `120` retained trace hashes match the unchanged indexes. The corrected Markdown is reproducible from those JSON reports. Deterministic v2 receipt hashes are `e1f6466a…d812`, `5a5a996c…cecc`, and `c6dab6cb…da7f`; telemetry is separate. Focused harness/Phase B/V2-input validation is `114 passed`. Full validation is `1462 passed, 18 skipped, 1 deselected, 44 failed` in `119.35 s`; the `44` failed node IDs exactly match `artifacts/bootstrap_tqc_humanoid/sandbox_baseline_failures_03a3.txt`. The sole deselection was the forbidden reward-lane receipt test, which was not executed or changed. Claim ceiling: interface and integrity repairs only, not behavioral evidence.

Fable resume: verify the immutable Phase A hash test, strict prior/provenance tamper tests, deterministic receipt/telemetry split, recovery bounds, and Phase B `task_inputs_v2.py` source binding; then commit this complete E003R1 slice without running the forbidden reward receipt, a simulator, or training.

Fable launch note (2026-09-05T19:32Z): builder `sol-builder-20260905-ft1` (packet `TASK-20260905-FT1`) starts from the clean main head after the strategy commit; read-only reviews `sol-review-sci-20260905-e003` and `sol-review-adv-20260905-e003` are running on the phase A commits. Fable resume: read the FT1 final, run the outside-sandbox suite, commit slice and integration separately, fold the two reviews, then packet `FT2` (training worker, supervision, persistence, report v2, CLI train); the disposable smoke needs a mailbox reservation; the five-seed cohort needs Samuel's authorization.

FT1 completion (2026-09-05T20:20:41Z): source-bound E1 receipt `c602e14a…f3788`, static phase-transfer receipt `eee389c5…b7953`, and run manifest `28d800f0…f9e0e` validate. Focused/adjacent tests are `56 passed`; the full suite is `1296 passed, 11 skipped, 1 deselected, 44 failed`, with all `44` failure node IDs exactly matching `artifacts/bootstrap_tqc_humanoid/sandbox_baseline_failures_03a3.txt`. The sole deselection was the forbidden reward-lane receipt test `tests/experiments/test_reward_target_speed_manifest.py::test_recorded_no_learning_runtime_receipt_replays_exactly`. Ruff lint, the `266`-file format check, and `git diff --check` pass. Fable resume: review the FT1 code and receipts, run the suite outside the builder sandbox if desired, commit the implementation slice, and launch FT2 without training until its reservation boundary is satisfied.

FT1 30-minute checkpoint (2026-09-05T20:02:41Z): lease and expert hash remained valid; `53` focused/adjacent tests, lint, and the 200-step real no-learning smoke passed. Full-suite validation was in progress. The exact forbidden reward-lane test is `tests/experiments/test_reward_target_speed_manifest.py::test_recorded_no_learning_runtime_receipt_replays_exactly`; its receipt was not regenerated.

Fable integration note (2026-09-05T19:11Z): Slice E003C2 committed as `74d7aa5`; Experiment
003 phase A is closed (60/60 episodes across the three tested step-300
running-speed handovers fell; alternative phases, timings, and
state-conditioned handovers remain untested). Astra accepted the convergence terms, merged main
`2fb31dc` into its lane as `8a48f32`, and is building a data-only target-speed
formula family; Fable answered the adapter question: reward-study task T2 holds
3.0 m/s on the fine-tuning runtime with the stock COM `x_velocity` under a
versioned input contract v2. Fable resume: read the fine-tuning runtime survey
final (`sol-survey-20260905-ft`), write the phase B runtime packet, launch it as
the writer; integrate Astra's reviewed merge after their M1 acceptance; the
first training beyond a 20-minute interface-check smoke needs a mailbox
reservation and Samuel's authorization.

Fable resume: verify the cycle-2 oracle's byte identity and provenance, `20/20` replay receipt, six-row report, exact `single_fast` outcome/behavior match, phase-A closure, ignored trace index, focused regression, and claim ceiling; then commit E003C2 without editing the frozen task, oracle, or reports.

- Updated: `2026-09-06T04:05Z` (FT2R3 isolation and mechanism-negative handoff)
- Repository: `/Users/samueldoane/Documents/ChatGPT/humanoid-harness`
- Precondition HEAD: `6d71e8b` (clean at FT2R3 launch; no Git writes)
- Git writes: none; Fable owns review and commit
- Write lease: `CLAIMED` by `sol-builder-20260906-ft2r3`, model `gpt-5.6-sol`, role `builder`, exact packet scope; launcher owns renewal and release
- Evidence class: `interface_check`; no training or behavioral evidence
- Builder wall time: `101m` through full-suite baseline reconciliation, final lint, authorized-scope format, diff, seals, and handoff

## Historical Experiment 003 phase-A handoff

The sections below preserve the pre-review phase-A handoff. E003R1's corrected
claim and evidence-chain status above supersedes its wording where they differ.

### Frozen inputs

| artifact | SHA-256 | frozen fact |
|---|---|---|
| v2 corpus manifest | `a3e9a7234b67194f0d4ac8d3961e9040f217ec9768e27451c279105aa3391090` | admitted source corpus |
| v2 E3 certificate | `5a91a265e057e6f8c6497d50a4a20ac3baab40b67c101115bc96b6f4b7555b74` | branch fall counts source |
| v2 validation manifest | `5eeedc84d13bf83fde534423fb5eb898160f543ef7dcdf325a7e136adbcfc434` | receipt-chain source |
| Experiment 003 library manifest | `ad57578dc2ed4fe3707da74a4f86620e758b1aa30064016785d187a5e86d3207` | raw file bytes |
| Experiment 003 task spec | `edb2cffde9f2eb087667d182f3da199ef6956aeadc6d45e9e218649d6fa9cbd7` | raw file bytes; frozen before evaluation |

Library speed statistics use pooled per-step root-`x` sidecar differences divided by the `0.015 s` control period across the `28` v2 clips whose E3 blocks passed.

| behavior | import receipt SHA-256 | strict NPZ SHA-256 | equivalence receipt SHA-256 | median speed (m/s) | IQR (m/s) | E3 falls / 1,000 steps |
|---|---|---|---|---:|---:|---:|
| expert | `b790f06ccb66ca45809eaa1b0c5cd6804e072c6fd9eb48c61838ee2e558bd9d7` | `60987a4e054db2e04f9cb3ab73e13dfe8e2f3ec7dec46346d2b9d0277ad18d9b` | `65b2090b783e381e799e90e372308018d4e9a50fa6be2df7934af5f24e78a23e` | `5.5207686161` | `0.7193024104` | `0.0555555556` |
| medium | `b1c07c2f5070b48ff6bb28983b16ed16d1616e7ad18f557639dad89bb1f4f172` | `2677ebb70cd20e0ba8f8a591814fc853e325277db65184ebafb5bf5e21198b04` | `ede73be8db0ec3411dfb1a5ce2dbbe109d49432d1aeba97e6e4b9bf4fb44d487` | `3.0740198759` | `0.4913530375` | `0` |
| simple | `bef2a2678d30f13a9e3350bf8c8f591661bb129b063276a15b8051f94ee313f7` | `b09aa921316640024e9703671d328d7ecbabe76f04cfed8c1d8fb86fbd95a917` | `604db637d316d3b8e46fb6f5d7a9b9004cfced267836394d89df183c87615268` | `0.8853599909` | `0.2107452405` | `0.1944444444` |

`simple` is the frozen slow actor because its median is `4.6354086253 m/s` below the expert, versus `2.4467487402 m/s` for `medium`. The target is expert speed on steps `[0,300)`, simple speed on `[300,600)`, and expert speed on `[600,1000)`.

### Cycle-0 result

| arm | median mean absolute speed error (m/s) | falls / 20 | median switches | median behavior time (steps) | median descriptive task return |
|---|---:|---:|---:|---|---:|
| `single_fast` | `2.0741721755` | `0` | `0` | expert `1,000` | `10,648.5529837515` |
| `single_slow` | `3.3071158305` | `0` | `0` | simple `1,000` | `5,772.1091802316` |
| `playback` | `3.1803425853` | `20` | `2` | expert `700`; simple `300` | `2,836.4346835564` |
| `handwritten` | `3.2062160613` | `20` | `2` | expert `972`; medium `28`; simple `0` | `2,954.1679253263` |

- `playback` switched exactly at boundaries `300` and `600` in every episode.
- The primary run produced `80` traces; replaying `single_fast` on all `20` seeds produced identical trace hashes.
- Evaluation wall time: `21.3701 s` recorded inside the report; `21.86 s` process wall including startup and writes. Determinism replay consumed `3.2953 s` of the recorded total.
- Runtime fingerprint SHA-256: `186fd2f4aa6b9ed9a0eb3de73faa0d2bbcb392a4fe52f992b6b443fbf33d2621`; it records plain `Humanoid-v5`, `terminate_when_unhealthy=false`, `TimeLimit=1000`, `0.015 s` control period, wrapper stack, model hash, dependency versions, and source hashes.
- Canonical report JSON SHA-256: `3d32ddcc3869f9a05df9beb1bc1f7816996bf3bda166eb4ca26f3ec9556aa623`.
- One-table report Markdown SHA-256: `cdbe252277b62749e09e029033077795e235c5cac208ce76f983ae23a98c68a0`.
- Ignored trace index: `artifacts/experiments_003/cycle_0/content_index.json`, `80` entries, SHA-256 `041a81cac8d77d717aacb31e6e633fcae2a47e09457fc92eade79dc103e7f983`; the ignored cycle directory is `54M`.

### Cycle-1 designer input

| artifact | SHA-256 | binding |
|---|---|---|
| `cycles/cycle_1/designer_prompt.md` | `53b319c5975fcd0cc5a54a2de3358cefe8a02e345982c2d89d6a5ac2ffde3010` | task, library, allowed schema/signals, rules, and cycle-0 report table |
| `cycles/cycle_1/expected_inputs.json` | `dc76c192e07bdd735df72f0263975e69ee378c8f6a95acd060d00383eb4d0351` | binds task, library, prompt, and prior report hash |

No steering text was supplied. The audit log records only this prompt read;
that observation is not an OS-enforced read allowlist.

### Cycle-1 candidate and result

| item | value |
|---|---|
| designer | `sol-designer-20260905-e003d1`; run `.orchestration/sol-runs/20260905T183122Z-b0788053-4fd7-4887-af1d-8c2ef411c93b`; audit log records the prompt read, not OS-enforced isolation |
| oracle raw SHA-256 | `32bfc555ffc578d4cf8f75823acc1ba98db6621b2bcd0204f0725b0bf70af029`; source and frozen copy matched byte-for-byte |
| oracle canonical SHA-256 | `9d0929cb785429776283e15765bdb3c3f19c60f1a0a3166317909f5f7bef381a` |
| contract validation | pass; no oracle edit |
| median MAE / falls | `3.2215072393 m/s / 20 of 20` |
| median switches | `3`; `14` episodes had `3`, `6` had `1` |
| median behavior time | expert `299`; medium `701`; simple `0` steps |
| slow-third fraction | expert `0`; medium `1`; simple `0` in every episode |
| median descriptive task return | `2,995.0573738434` |

- Every episode switched `expert -> medium` at step `300` and first fell at boundary `322`-`357`, within `100` steps.
- The candidate matched `playback` and `handwritten` at `20` falls but worsened median MAE by `0.0411646540 m/s` and `0.0152911780 m/s`, respectively.
- The candidate worsened both metrics versus `single_fast`; versus `single_slow`, falls worsened and median MAE improved by `0.0856085912 m/s`. No combined ranking was predeclared.
- The required per-switch and per-episode slow-third tables are in `cycles/cycle_1/report_1.md`; provenance is in `cycles/cycle_1/cycle_record.md`.

| receipt | SHA-256 / result |
|---|---|
| runtime fingerprint | `186fd2f4aa6b9ed9a0eb3de73faa0d2bbcb392a4fe52f992b6b443fbf33d2621`; unchanged from cycle 0 |
| `report_1.json` | `480d5b3521ce66593e0258f3f42904f969c72b83141b330bc7ff4759144c25f0` |
| `report_1.md` | corrected render `116e2a836ea5f2a0ae6b596e7005c0e0dbee701ad7a75d468c3d73d92817207b`; pre-review bytes read by cycle 2 were `88cefeeeb5cfe47f40d4623233b3a76a73e18f7ea0b70a7b186369b04029170a` |
| ignored trace index | `4d2c419140b71d1cc13f04c6af802d9fb569aaf690621cc17cd99ab5f4057097`; `20` entries; `14M` cycle directory |
| determinism | `20/20` replay hashes matched; `7.4264 s` report wall |
| evaluation wall | `15.0420 s` in report; `15.53 s` process wall |

### Cycle-2 designer input

| artifact | SHA-256 | binding |
|---|---|---|
| `cycles/cycle_2/designer_prompt.md` | `99b1714912970a98d9436dfbbf9dee2f975e32d574d8a36d81ff085d02c77dab` | task, library, five-row prior report, rules, and no human steering text |
| `cycles/cycle_2/expected_inputs.json` | `4048d6477c7864c838c5d87717c016827d5e62dc8c8569a0a2977e77702c6d6b` | binds task, library, prompt, and `report_1.json` SHA-256 `480d5b...25f0` |

The read-only designer's launch packet supplied steering from Fable, the
orchestrating agent; no human steering was supplied. The exact required text is
quoted in `cycles/cycle_2/cycle_record.md`.

### Cycle-2 candidate and result

| item | value |
|---|---|
| designer | `sol-designer-20260905-e003d2`; run `.orchestration/sol-runs/20260905T185111Z-78f29dc3-9fba-4f26-a393-b86bc71f4349`; audit log records the two declared reads, not OS-enforced isolation |
| oracle raw SHA-256 | `f1cbc2784783d4f34312b036a911c536f8406aa3da1ffcf52f35f4927808173a`; source and frozen copy matched byte-for-byte |
| oracle canonical SHA-256 | `5868d09ea71a4fcd5f1be46ef92fa36c83ac7e3867222cecb9fa7f3000e835bf` |
| contract validation | Pass; no oracle edit |
| median MAE / falls | `2.0741721755 m/s / 0 of 20` |
| median switches | `0` |
| median behavior time | expert `1,000`; medium `0`; simple `0` steps |
| slow-third fraction | expert `1`; medium `0`; simple `0` in every episode |
| median descriptive task return | `10,648.5529837515` |

- The program holds `expert` throughout and is behaviorally identical to
  cycle-0 `single_fast`. All comparable outcome fields and normalized per-step
  behavior projections matched byte-for-byte for `20/20` seeds.
- Full serialized report rows differ only where expected for provenance,
  trace location, size, and hash, wall time, and cycle-2 diagnostics.
- The candidate passed the never-fall requirement but did not execute the
  requested slow third. It is not an oracle-improvement result.

| receipt | SHA-256 / result |
|---|---|
| runtime fingerprint | `186fd2f4aa6b9ed9a0eb3de73faa0d2bbcb392a4fe52f992b6b443fbf33d2621`; unchanged from cycles 0 and 1 |
| `report_2.json` | `e677b9f3c3daabdb13648a18981c39b47bc74fa2e6908e7fe1af9d73674e194c` |
| `report_2.md` | corrected render `c071106fe3bbfc050ba1fdcba234a1da32f0d43df365851e5360c54b102420b6` |
| ignored trace index | `a210e04509a1ac2266981c1ea0af1407ce5059d72d633d6ba62daf9970bdf775`; `20` entries; `13M` cycle directory |
| determinism | `20/20` replay hashes matched; `3.3180 s` report wall |
| evaluation wall | `6.8207 s` in report; `7.30 s` process wall |

### Phase-A closure

None of the three tested step-300 running-speed handovers survived; alternative
phases, timings, and state-conditioned handovers remain untested. Those three
designs (`playback`, `handwritten`, and cycle 1) fell in `60/60` episodes. An
admitted deceleration behavior with entry coverage at expert running speeds
and a validated safe handoff to `medium` or `simple` remains one untested
direction. The full table and claim boundary are frozen in
`experiments/003_composition_speed_profile/PROTOCOL.md`.

Phase B keeps task T1 and replaces controller switching with a tracker-following
fine-tuning runtime warm-started at the expert. The runtime learns transitions
from the composed reference. No cycle-3 prompt was prepared.

### Validation

| check | result | wall time |
|---|---|---:|
| focused harness tests | `14 passed` | `0.59 s` process wall |
| cycle-2 report regression | Four cycle-0 arms plus cycle-1 and cycle-2 candidates; prior-report binding, source labels, and malformed-history rejection | included in focused tests |
| Ruff lint | pass | `0.01 s` |
| Ruff format check | `252 files already formatted` | `0.01 s` |
| fake-runtime CLI smoke | cycles 0, 1, and 2 at `2` episodes × `50` steps; pass | included in focused tests |
| playback transition regression | switches exactly at `300` and `600`; pass | included in focused tests |
| metric independence regression | corrupting a written trace does not alter the in-memory recomputed metric; pass | included in focused tests |
| independent cycle-2 audit | Canonical report, exact seeds, all `20` indexed traces, runtime binding, and `single_fast` outcome/behavior identity; pass | `0.32 s` |
| final `git diff --check` | pass | `0.01 s` |

The only code change fixes the cycle-2 report carry-forward path. Previously it
discarded the cycle-1 candidate and mislabeled the current row as cycle 1; the
focused regression failed before the fix and now covers three cycles. No task
specification, oracle, actor, runtime, reward, seed, horizon, or metric changed.

### Claim ceiling

Phase A supports only that three exploratory controller-switching cycles ran on
the frozen plain `Humanoid-v5` runtime over the predetermined seeds, their
canonical reports exist, and their determinism replays matched. `task_return`
is descriptive stock reward. Controller switching stood in for tracker
following. These results support no oracle-quality, generalization,
frozen-tracker, reference-following, task-reward, naturalness, robustness, or
humanoid-competence claim.
