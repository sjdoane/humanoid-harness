# Current research handoff

| status | current truth |
|---|---|
| progress | `TASK-20260905-E003C0` completed its one-shot cycle-0 evaluation: four predeclared controller-switching arms ran on seeds `97001`-`97020` for `1,000` steps, all `20/20` deterministic replay hashes matched, `report_0` exists, and the cycle-1 designer prompt is prepared. |
| bottleneck | Both switching arms fell in `20/20` episodes. The handwritten arm spent a median `28` steps in `medium`, then entered recovery and spent `0` steps in `simple`; no tuning or rerun followed. Controller switching remains only a stand-in for frozen-tracker reference following. |
| next step | Fable reviews and commits this builder slice; after that, a separate read-only designer may receive only `cycles/cycle_1/designer_prompt.md` and produce one cycle-1 oracle. Do not infer oracle quality or change the frozen task from cycle-0 results. |

Fable integration note (2026-09-05T18:33Z): Slice E003C0 committed as `50fd60c`; the
read-only designer `sol-designer-20260905-e003d1` is producing the cycle-1
oracle from `cycles/cycle_1/designer_prompt.md` alone; packet `E003C1` then
evaluates it. Astra's counterproposal (cycle-first convergence, data-only
reward family first, merge main into Astra then merge back) is accepted in the
mailbox. Fable resume: read the designer final, write
`.orchestration/oracles/cycle_1_candidate.json`, launch `E003C1`, then record
cycle 1 in the strategy; next packet after that is the fine-tuning runtime.

Fable resume: verify the hashes, report, ignored trace index, exact-baseline test receipt, and claim ceiling below; then commit the reviewable E003C0 slice without rewriting the frozen task or oracle arms.

- Updated: `2026-09-05T18:25:44Z`
- Repository: `/Users/samueldoane/Documents/ChatGPT/humanoid-harness`
- Precondition HEAD: `a0a0a3bc3560dccf95519654d72fefbbb706ecda` (clean before this slice; at required `a0a0a3b`)
- Git writes: none; Fable owns review and commit
- Write lease: `CLAIMED` by `sol-builder-20260905-e003c0`, model `gpt-5.6-sol`, role `builder`, exact authorized scope; launcher owns renewal and release
- Evidence class: `exploratory_oracle_cycle`

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

## Cycle-1 handoff

| artifact | SHA-256 | binding |
|---|---|---|
| `cycles/cycle_1/designer_prompt.md` | `53b319c5975fcd0cc5a54a2de3358cefe8a02e345982c2d89d6a5ac2ffde3010` | task, library, allowed schema/signals, rules, and cycle-0 report table |
| `cycles/cycle_1/expected_inputs.json` | `dc76c192e07bdd735df72f0263975e69ee378c8f6a95acd060d00383eb4d0351` | binds task, library, prompt, and prior report hash |

No steering text was supplied. The next designer is not this builder and sees only the prompt.

## Validation

| check | result | wall time |
|---|---|---:|
| focused harness tests | `13 passed` | `0.62 s` process wall |
| full suite with the forbidden reward-lane test deselected by exact node ID | `1,267 passed, 44 failed, 11 skipped, 1 deselected`; failed node-ID set exactly equals all `44` entries in `sandbox_baseline_failures.txt`, with no duplicate, missing, or unexpected ID | `113.65 s` process wall on the final instrumented run |
| Ruff lint | pass | `0.04 s` |
| Ruff format check | `247 files already formatted` | `0.03 s` |
| fake-runtime CLI smoke | `2` episodes × `50` steps; pass | included in focused tests |
| playback transition regression | switches exactly at `300` and `600`; pass | included in focused tests |
| metric independence regression | corrupting a written trace does not alter the in-memory recomputed metric; pass | included in focused tests |
| final `git diff --check` | pass | `<0.01 s` |

The final full-suite rerun wrote only `/private/tmp/e003c0-full-suite.xml` for mechanical comparison; it is not a repository artifact. An earlier pre-gate collection collision caused by a duplicate test-module basename was corrected within this slice before the final focused and full gates.

## Claim ceiling

Passing supports only that cycle 0 ran on the frozen runtime with four predeclared oracle programs, its canonical report exists, deterministic replay matched for the checked arm and seeds, and the cycle-1 designer prompt exists. `task_return` is descriptive stock reward. Controller switching stands in for tracker following. These results support no oracle-quality, generalization, frozen-tracker, reference-following, task-reward, naturalness, robustness, or humanoid-competence claim.
