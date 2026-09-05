# Current research handoff

| status | current truth |
|---|---|
| progress | `TASK-20260905-E003C1` evaluated the read-only designer's byte-exact cycle-1 candidate on seeds `97001`-`97020` for `1,000` steps; all `20/20` determinism replay hashes matched, the detailed report exists, and cycle 2 is prepared. |
| bottleneck | The candidate fell in `20/20`, never reached `simple`, and spent `100%` of `[300,600)` in `medium`. It matched both switching baselines' fall count and had higher median MAE (`3.2215072393 m/s`). Controller switching remains only a stand-in for frozen-tracker reference following. |
| next step | Fable reviews and commits this slice, records the failed candidate in strategy, and retains the prepared cycle-2 prompt. The already-planned fine-tuning-runtime packet remains the next write slice unless Fable explicitly schedules another read-only designer. |

Fable resume: verify the byte-identical oracle and provenance, `20/20` replay receipt, five-row report, transition and slow-third tables, ignored trace index, cycle-2 bindings, focused-test receipt, and claim ceiling; then commit E003C1 without editing the frozen task, oracle, or reports.

- Updated: `2026-09-05T18:48:39Z`
- Repository: `/Users/samueldoane/Documents/ChatGPT/humanoid-harness`
- Precondition HEAD: `2fb31dcc85a23d1d1ed0b8ab3c86b30d460d7d51` (clean before this slice; exact launch-note commit)
- Git writes: none; Fable owns review and commit
- Write lease: `CLAIMED` by `sol-builder-20260905-e003c1`, model `gpt-5.6-sol`, role `builder`, exact authorized scope; launcher owns renewal and release
- Evidence class: `exploratory_oracle_cycle`
- Builder wall time: `14m18s` from launcher start through the final lease, scope, and diff check

## Frozen inputs

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

## Cycle-0 result

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

## Cycle-1 designer input

| artifact | SHA-256 | binding |
|---|---|---|
| `cycles/cycle_1/designer_prompt.md` | `53b319c5975fcd0cc5a54a2de3358cefe8a02e345982c2d89d6a5ac2ffde3010` | task, library, allowed schema/signals, rules, and cycle-0 report table |
| `cycles/cycle_1/expected_inputs.json` | `dc76c192e07bdd735df72f0263975e69ee378c8f6a95acd060d00383eb4d0351` | binds task, library, prompt, and prior report hash |

No steering text was supplied. The read-only designer saw only this prompt.

## Cycle-1 candidate and result

| item | value |
|---|---|
| designer | `sol-designer-20260905-e003d1`; run `.orchestration/sol-runs/20260905T183122Z-b0788053-4fd7-4887-af1d-8c2ef411c93b`; prompt-only visibility |
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
| `report_1.md` | `88cefeeeb5cfe47f40d4623233b3a76a73e18f7ea0b70a7b186369b04029170a` |
| ignored trace index | `4d2c419140b71d1cc13f04c6af802d9fb569aaf690621cc17cd99ab5f4057097`; `20` entries; `14M` cycle directory |
| determinism | `20/20` replay hashes matched; `7.4264 s` report wall |
| evaluation wall | `15.0420 s` in report; `15.53 s` process wall |

## Cycle-2 handoff

| artifact | SHA-256 | binding |
|---|---|---|
| `cycles/cycle_2/designer_prompt.md` | `99b1714912970a98d9436dfbbf9dee2f975e32d574d8a36d81ff085d02c77dab` | task, library, five-row prior report, rules, and verbatim steering sentence |
| `cycles/cycle_2/expected_inputs.json` | `4048d6477c7864c838c5d87717c016827d5e62dc8c8569a0a2977e77702c6d6b` | binds task, library, prompt, and `report_1.json` SHA-256 `480d5b...25f0` |

Steering source: Fable launch note; no human supplied steering. Verbatim text: `No additional steering text was supplied.`

## Validation

| check | result | wall time |
|---|---|---:|
| focused harness tests | `13 passed` | `0.59 s` process wall |
| cycle-1 report regression | four frozen baseline rows plus candidate; switch speed and both fall-window paths; slow-third fractions; cycle-2 prompt binding | included in focused tests |
| Ruff lint | pass | `0.01 s` |
| Ruff format check | `250 files already formatted` | `<0.01 s` |
| fake-runtime CLI smoke | cycles 0 and 1 at `2` episodes × `50` steps; pass | included in focused tests |
| playback transition regression | switches exactly at `300` and `600`; pass | included in focused tests |
| metric independence regression | corrupting a written trace does not alter the in-memory recomputed metric; pass | included in focused tests |
| final `git diff --check` | pass | `0.01 s` |

The E003C1 packet required focused tests, lint, format, and `git diff --check`; it did not request another full-suite run. The E003C0 exact-baseline full-suite receipt remains at base commit `2fb31dc`. The only code change fixes the report path that otherwise omitted the frozen cycle-0 comparison, per-switch diagnostics, slow-third fractions, and explicit per-metric outcomes; the regression covers the positive and negative fall-window paths.

## Claim ceiling

Passing supports only that the byte-exact cycle-1 candidate ran on the frozen plain `Humanoid-v5` controller-switching runtime over the predetermined seeds, its canonical report and diagnostics exist, its determinism replay matched, and the cycle-2 prompt is prepared. `task_return` is descriptive stock reward. Controller switching stands in for tracker following. These results support no oracle-quality, generalization, frozen-tracker, reference-following, task-reward, naturalness, robustness, or humanoid-competence claim.
