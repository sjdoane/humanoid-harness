# Reference corpus v1 protocol and stopped run

| status | current truth |
|---|---|
| progress | The three frozen actors produced all `108` corpus clips. A separate process verified every one of their `108,000` transitions, and E3 qualified `27/36` blocks. |
| bottleneck | The first expert development-screen attempt, seed `96001`, stopped on the plain-versus-instrumented equivalence canary before a screen bundle was published. |
| next step | Preserve this run as stopped. Diagnose the canary operation asymmetry and authorize a new versioned run before any screen seed is executed again. |

## Status and claim ceiling

`TASK-20260905-03BRUN2` is **stopped and incomplete**. The retained evidence
supports only the `108` named same-runtime corpus trajectories and their
full-clip replay certificates, plus the reported E3 fork-identifiability
result. It does not support a development-screen result or a completed
`reference_corpus_v1` run.

No tracker, E4, E5, oracle, naturalness, robustness, or general task-composition
claim is made. The three actor continuations are a first composition benchmark
with distinct admitted segments and transition evidence.

## Frozen design

| item | frozen value |
|---|---|
| runtime | Project `Humanoid-v5`, CPU deterministic actor mean, `terminate_when_unhealthy=false`, `1,000` control steps, `15 ms` controls over five `3 ms` substeps |
| corpus | `36` common-reset blocks; seeds `120001-120012`, `120101-120120`, and `120201-120204`; actors `expert`, `medium`, `simple`; no replacement seeds |
| screen | Expert seeds `96001-96020`; `1,000` steps; four direct-state gates; reward only a plain-versus-instrumented equivalence canary |
| replay | Separate-process, all-transition Tier-D check; any transition mismatch fails the clip |
| E3 | Expert-versus-medium and expert-versus-simple same-state forks with the thresholds in `e3_manifest_v1.json` |
| retry rule | Failed attempts remain in the ledger; this stopped run is not retried |

## Reset-free actor preflight

All six receipts loaded as canonical JSON followed by exactly one terminating
LF. Canonical JSON without the LF and with two LFs are rejected by the runner.
All three NPZ hashes matched `PUBLIC_EXPERT_IMPORT.md` before any reset.

| actor | import receipt SHA-256 | strict NPZ SHA-256 | equivalence receipt SHA-256 |
|---|---|---|---|
| expert | `b790f06ccb66ca45809eaa1b0c5cd6804e072c6fd9eb48c61838ee2e558bd9d7` | `60987a4e054db2e04f9cb3ab73e13dfe8e2f3ec7dec46346d2b9d0277ad18d9b` | `65b2090b783e381e799e90e372308018d4e9a50fa6be2df7934af5f24e78a23e` |
| medium | `b1c07c2f5070b48ff6bb28983b16ed16d1616e7ad18f557639dad89bb1f4f172` | `2677ebb70cd20e0ba8f8a591814fc853e325277db65184ebafb5bf5e21198b04` | `ede73be8db0ec3411dfb1a5ce2dbbe109d49432d1aeba97e6e4b9bf4fb44d487` |
| simple | `bef2a2678d30f13a9e3350bf8c8f591661bb129b063276a15b8051f94ee313f7` | `b09aa921316640024e9703671d328d7ecbabe76f04cfed8c1d8fb86fbd95a917` | `604db637d316d3b8e46fb6f5d7a9b9004cfced267836394d89df183c87615268` |

The frozen E3 manifest SHA-256 is
`003b5b045ca9f8af2b0af3f559af206e1000835deea98ff98bd45a79fa5b9504`.

## Run receipts

| artifact | SHA-256 | result |
|---|---|---|
| E1 initialization-identity receipt | `28725ecfba2ca89f4b4608e5e8d5b5024cfe2ed8df71383bc03a248891edf266` | precondition pass |
| sandbox baseline failure IDs | `c06586e4449fad6b00e6356e6a70e2d75e3476fb8f4fbbf0b2c4e72c547a8345` | `44` expected IDs |
| corpus manifest | `87c25d9b380605556a62681e44879943275a358b7136b322e0b103f2b8dff9f5` | `108` clips in frozen order |
| attempt ledger | `29c03cbbd89cbe3045021e54bd2edb6e0a47f8624d152aefa22f01377e24c5e6` | `108` collected; seed `96001` started; stopped event retained |
| runner log | `03d85f84bb6ebb06c6fe15ec4aa4fb9262895d3525158f482bb45844a680f6f2` | exit `1`; `61.57 s` wall time |
| corpus-only certifier request | `05b22aadc791b873d7e3d8aa17558b154d200603d13e5341a2174b9a67f0af51` | `108` immutable bundles |
| corpus-only certifier log | `30c87b8977a0cc0aca1a8c6cf3ed2d0544491fd92befe9c0f2d7ac9d3248b4c9` | exit `0`; `64.26 s` wall time |
| stopped-run Tier-D aggregate | `3b80d2b3afa5a3206004acd5cc3507cfc4483e3feac6254545ce8a395fc4d616` | `108/108` pass; binds every per-clip certificate |
| E3 fork certificate | `c977f5068e7c71bcd75a91da3ccf5fc5825e63f3cad4d1608f4106ba64be50fa` | `27/36` blocks pass; `61/72` pairs pass |
| stopped-run content index | `7d9bc501f77466f7dc77151812c8024e6050e6ceda985a9e286a49628387acbb` | partial evidence only |
| payload-free validation manifest | `aa17751cb665afef0cd4d8f9a353e7889bb227f3819de98c5de9a92db48a3646` | binds source tree, commands, counts, runtime, skips, and all `108` certificate hashes |

The completed-run aggregate, completed-run content index, run result, and
`public_expert_development_screen_v1.json` do not exist. The stopped-run
aggregate contains all `108` per-clip bundle, reference-identity, and Tier-D
certificate hashes.

## Tier-D results

| actor | clips | passed | failed | transitions verified |
|---|---:|---:|---:|---:|
| expert | 36 | 36 | 0 | 36,000 |
| medium | 36 | 36 | 0 | 36,000 |
| simple | 36 | 36 | 0 | 36,000 |
| **total** | **108** | **108** | **0** | **108,000** |

## E3 results

`P` means the locked pair or block rule passed; `F` means it failed. Pair
failures were caused only by the predeclared branch
`horizon/collapse/contact/replay` conjunction. No failed pair missed the action
or future-variation thresholds.

| seed | expert vs medium | expert vs simple | block |
|---:|:---:|:---:|:---:|
| 120001 | P | P | P |
| 120002 | F | P | F |
| 120003 | P | P | P |
| 120004 | P | P | P |
| 120005 | P | P | P |
| 120006 | P | P | P |
| 120007 | P | P | P |
| 120008 | F | F | F |
| 120009 | P | F | F |
| 120010 | P | P | P |
| 120011 | P | P | P |
| 120012 | P | P | P |
| 120101 | P | P | P |
| 120102 | F | F | F |
| 120103 | P | F | F |
| 120104 | P | P | P |
| 120105 | P | F | F |
| 120106 | P | P | P |
| 120107 | P | P | P |
| 120108 | P | P | P |
| 120109 | P | P | P |
| 120110 | P | P | P |
| 120111 | P | F | F |
| 120112 | P | P | P |
| 120113 | P | P | P |
| 120114 | P | P | P |
| 120115 | P | P | P |
| 120116 | P | P | P |
| 120117 | P | P | P |
| 120118 | F | P | F |
| 120119 | P | F | F |
| 120120 | P | P | P |
| 120201 | P | P | P |
| 120202 | P | P | P |
| 120203 | P | P | P |
| 120204 | P | P | P |

| E3 summary | pass | fail |
|---|---:|---:|
| blocks | 27 | 9 |
| expert vs medium pairs | 32 | 4 |
| expert vs simple pairs | 29 | 7 |
| expert branches | 34 | 2 |
| medium branches | 34 | 2 |
| simple branches | 31 | 5 |

## Development-screen result

| field | result |
|---|---|
| attempted | seed `96001`, reset order `108` |
| completed clips | `0/20` |
| stop | `ReferenceCorpusContractError: plain/instrumented reward canary differs` |
| four locomotion gates | not computed |
| screen pass/fail | no result |
| receipt | absent |

The error combines returned-observation, reward, and flag equality in one
guard, so the retained exception does not identify which subfield differed.
The leading code-level diagnosis is an operation asymmetry: reset already
calls `mj_forward`, then `_capture_boundary` calls it again only on the
instrumented environment before the first action. This diagnosis is not a
measured fix. No seed was rerun and no threshold was tuned.

## Validation

| check | result |
|---|---|
| receipt-loader regression | `1 passed` |
| focused corpus tests | `29 passed in 1.06 s` |
| full suite with reward-lane replay test deselected by exact name | `1,250 passed, 44 failed, 11 skipped, 1 deselected in 114.24 s`; failing IDs exactly equal the saved sandbox baseline |
| Ruff lint | pass |
| Ruff format | `231` files formatted |
| `git diff --check` | pass |
