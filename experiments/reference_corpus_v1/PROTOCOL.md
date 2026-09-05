# Reference corpus protocol: v1 superseded, v2 completed

| status | current truth |
|---|---|
| progress | Run v2 collected `108/108` corpus clips and `20/20` screen clips. All `128,000` transitions passed separate-process replay and per-step plain-runtime comparison; E3 qualified `28/36` blocks. |
| bottleneck | The exact imported expert failed the frozen development screen: `19/20` resets remained healthy and upright; the velocity and displacement gates passed. No retry or tuning occurred. |
| next step | Fable reviews and commits the v2 slice, records the failed development screen, and preserves v1 as superseded evidence rather than using either run for a tracker, E4, E5, oracle, naturalness, or robustness claim. |

## Status and claim ceiling

`TASK-20260905-03BFIX` run v2 is **complete with a failed development screen**.
The retained evidence supports only the named clips as plain-runtime
closed-loop trajectories of the exact imported actors, their full-clip replay
certificates and per-step plain-comparison receipts, the reported qualifying E3
fork corpus, and the exact imported expert's failed screen in the pinned local
runtime.

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
| retry rule | Failed attempts remain in the ledger; v1 was not retried, and v2 uses the same frozen seeds with no replacement or screen retry |

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

## Run v2 results

Run v2 uses replay method
`predecessor_transition_cache_rebuild/v2`. For transition `t > 0`, the
certifier restores boundary `t-1`, executes and discards the stored predecessor
transition, verifies boundary `t`, then evaluates the target transition. It
verifies the reset boundary directly for `t = 0`.

### Run v2 receipts

| artifact | raw SHA-256 | result |
|---|---|---|
| E1 initialization-identity receipt | `28725ecfba2ca89f4b4608e5e8d5b5024cfe2ed8df71383bc03a248891edf266` | precondition pass |
| sandbox baseline failure IDs | `c06586e4449fad6b00e6356e6a70e2d75e3476fb8f4fbbf0b2c4e72c547a8345` | `44` expected IDs |
| frozen E3 manifest | `003b5b045ca9f8af2b0af3f559af206e1000835deea98ff98bd45a79fa5b9504` | unchanged design |
| corpus manifest | `a3e9a7234b67194f0d4ac8d3961e9040f217ec9768e27451c279105aa3391090` | `108` corpus clips in frozen order |
| attempt ledger | `2600d408122abb5d2f2d981a75241ff7fc8dea7db2e6a50a5a0e61f4939bdf5d` | `128` one-shot attempts; no replacement seeds |
| runner log | `2d37545f65482d8bec1566a983ed90ca8f01cc6d4814ed57018451c338423008` | exit `0`; `291.64 s` wall time |
| Tier-D certifier request | `cc2d70800c59828716950365da678f85c2b04422d797ee859e828672c94acac4` | `128` bundles; replay method v2 |
| Tier-D aggregate | `dcf06c8e96ebd4d7c3171ab6d3062a73045addcb1be992e0e2d3ca5f62801bf7` | `128/128` pass; internal result `d70183390664cff07730814c72cb34aec668b324060f0d4978a0175b6d24e584` |
| E3 fork certificate | `5a91a265e057e6f8c6497d50a4a20ac3baab40b67c101115bc96b6f4b7555b74` | `28/36` blocks; internal result `eaeab5fb7a06b9419233b553c5fc70eca16eb62fe6d119b842c86291a306ae4d` |
| expert screen receipt | `0ff4bd14712729ca2bdced8d5c8511317ab0f118a98adce6f75557e24eff5a60` | failed; internal result `ecd9f07077ddf29772008b8e8ff5676609e75fbd893c9a41b4d27e2dcd577c03` |
| content index | `9347c741241b1138ce361f01ad319d05e09a7597c9ef51d9715b86dee6c34afa` | `128` clips and `83` immutable objects; internal content `c3cf17100248a05bc609d28ff4a7bb7fbc55b61ffb6d1d6a2a4e34b23b2b6ba8` |
| run result | `c41d78ff0c1d2fdd899a53cf419b0b0d3f43b8d9d24e072cca400af42e6f1e98` | completed; screen false |
| full-suite log | `ebff2d7543398816dcd6fec88858a9313f2ccb82ebe2f322ac24cd231b873f00` | exact sandbox baseline |
| payload-free validation manifest | `5eeedc84d13bf83fde534423fb5eb898160f543ef7dcdf325a7e136adbcfc434` | binds all `128` bundle, payload, reference, plain-comparison, and Tier-D receipts |

### Run v2 replay and plain-comparison results

| clip group | actor | clips | Tier-D pass/fail | plain comparison pass/fail | transitions verified and compared |
|---|---|---:|---:|---:|---:|
| corpus | expert | 36 | 36 / 0 | 36 / 0 | 36,000 |
| corpus | medium | 36 | 36 / 0 | 36 / 0 | 36,000 |
| corpus | simple | 36 | 36 / 0 | 36 / 0 | 36,000 |
| screen | expert | 20 | 20 / 0 | 20 / 0 | 20,000 |
| **total** |  | **128** | **128 / 0** | **128 / 0** | **128,000** |

Each comparison checks, per boundary or transition as applicable, action bytes,
full integration state, explicit `qpos` and `qvel`, `cfrc_ext`, canonical and
returned observations, reward, simulation time, wrapper counter and flags, and
result flags. The payload-free validation manifest records every per-clip
bundle, payload, reference identity, comparison, boundary-receipt array,
transition-receipt array, Tier-D certificate, and replay-verification hash.

### Run v2 E3 results

`P` means the locked pair or block rule passed; `F` means it failed. All ten
failed pairs failed only the locked branch
`horizon/collapse/contact/replay` conjunction; none missed the action or
future-variation thresholds.

| seed | expert vs medium | expert vs simple | block |
|---:|:---:|:---:|:---:|
| 120001 | P | P | P |
| 120002 | P | P | P |
| 120003 | P | P | P |
| 120004 | P | F | F |
| 120005 | P | P | P |
| 120006 | F | F | F |
| 120007 | P | P | P |
| 120008 | P | P | P |
| 120009 | P | P | P |
| 120010 | P | F | F |
| 120011 | P | P | P |
| 120012 | P | P | P |
| 120101 | P | P | P |
| 120102 | P | P | P |
| 120103 | P | P | P |
| 120104 | P | P | P |
| 120105 | P | P | P |
| 120106 | P | P | P |
| 120107 | P | F | F |
| 120108 | P | P | P |
| 120109 | P | P | P |
| 120110 | P | P | P |
| 120111 | P | P | P |
| 120112 | P | P | P |
| 120113 | P | P | P |
| 120114 | P | P | P |
| 120115 | P | F | F |
| 120116 | P | P | P |
| 120117 | P | P | P |
| 120118 | P | F | F |
| 120119 | P | P | P |
| 120120 | P | P | P |
| 120201 | F | F | F |
| 120202 | P | P | P |
| 120203 | P | F | F |
| 120204 | P | P | P |

| E3 summary | pass | fail |
|---|---:|---:|
| blocks | 28 | 8 |
| expert vs medium pairs | 34 | 2 |
| expert vs simple pairs | 28 | 8 |
| expert branches | 34 | 2 |
| medium branches | 36 | 0 |
| simple branches | 29 | 7 |

The `62/72` qualifying pair count exceeds the locked minimum, so the corpus is
a qualifying E3 fork corpus.

### Run v2 development-screen result

| gate | required | observed | result |
|---|---:|---:|:---:|
| healthy full-horizon episodes | 20 / 20 | 19 / 20 | fail |
| upright full-horizon episodes | 20 / 20 | 19 / 20 | fail |
| median forward velocity | at least `0.5 m/s` | `5.001603770686775 m/s` | pass |
| episodes with at least `5 m` forward displacement | at least 18 / 20 | 20 / 20 | pass |

The exact imported expert therefore **failed** the predeclared screen. Seed
`96018` first became unhealthy at step `860` and first failed the upright gate
at step `871`; its minimum root height was `0.16328118194073535 m`, upright
fraction `0.87`, forward displacement `65.41267369164503 m`, average forward
velocity `4.360844912776202 m/s`, and non-foot floor-contact fraction `0.107`.
All 20 screen clips nevertheless passed full-clip replay and plain-runtime
comparison. No seed was replaced, rerun, or tuned.

### Run v2 validation

| check | result |
|---|---|
| focused corpus tests | `33 passed in 1.93 s` |
| ordered corpus plus TQC-runtime tests | `34 passed, 9 failed in 2.18 s`; all nine are the saved `cpu_model=arm` sandbox baseline |
| full suite with reward-lane replay test deselected by exact name | `1,254 passed, 44 failed, 11 skipped, 1 deselected in 117.11 s`; failing IDs exactly equal the saved sandbox baseline |
| Ruff lint | pass |
| Ruff format | `231` files already formatted |
| `git diff --check` | pass |

The runner's `291.64 s` wall time includes collection and the separate-process
certifier. A certifier-only wall timer was not retained. The reward-lane test
and forbidden receipt were not read or modified.

## Superseded v1 instrumented-schedule run

The following v1 results are preserved verbatim as historical evidence of the
instrumented schedule. They are superseded by v2 because the collector called
`mj_forward` after each live step and synthesized the next policy input from
the refreshed cache. V1 replay compared against that same helper and therefore
did not establish plain-runtime closed-loop trajectories.

### V1 run receipts

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

### V1 Tier-D results

| actor | clips | passed | failed | transitions verified |
|---|---:|---:|---:|---:|
| expert | 36 | 36 | 0 | 36,000 |
| medium | 36 | 36 | 0 | 36,000 |
| simple | 36 | 36 | 0 | 36,000 |
| **total** | **108** | **108** | **0** | **108,000** |

### V1 E3 results

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

### V1 development-screen result

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

### V1 validation

| check | result |
|---|---|
| receipt-loader regression | `1 passed` |
| focused corpus tests | `29 passed in 1.06 s` |
| full suite with reward-lane replay test deselected by exact name | `1,250 passed, 44 failed, 11 skipped, 1 deselected in 114.24 s`; failing IDs exactly equal the saved sandbox baseline |
| Ruff lint | pass |
| Ruff format | `231` files formatted |
| `git diff --check` | pass |
