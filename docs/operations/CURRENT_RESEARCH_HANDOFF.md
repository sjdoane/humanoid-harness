# Current research handoff

| status | current truth |
|---|---|
| progress | `TASK-20260905-E003R1` repaired the Experiment 003 interface and evidence chain: exact execution seal, recomputed library statistics, separated authority identities, deterministic chained scientific receipts plus telemetry, strict shared prior validation, report/trace schemas, designer provenance, bounded recovery, and Astra V2 input binding. No simulator ran outside tests; no training ran. |
| bottleneck | The repaired metric-core digest intentionally differs from the historical Phase A digest, so old arms cannot be carried into a new cycle without re-evaluation. R05, R06, and R10 remain explicitly deferred in `HARDENING_BACKLOG.md`; the full sandbox suite retains exactly its `44` recorded environment-bound failures. |
| next step | Fable reviews and commits this repair slice, preserving the Phase A JSON, trace, oracle, task, and library bytes. Any later experiment must start from a fresh sealed evaluation family; FT2 still requires its own packet and authorization boundaries. |

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

- Updated: `2026-09-05T21:38Z` (E003R1 validation handoff)
- Repository: `/Users/samueldoane/Documents/ChatGPT/humanoid-harness`
- Precondition HEAD: `8d91a81` (clean before E003R1; no Git writes)
- Git writes: none; Fable owns review and commit
- Write lease: `CLAIMED` by `sol-builder-20260905-e003r1`, model `gpt-5.6-sol`, role `builder`, exact authorized scope; launcher owns renewal and release
- Evidence class: interface and integrity repair; no behavioral evidence
- Builder wall time: `73m` through the baseline-matched full-suite validation; final lint and diff checks followed

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
