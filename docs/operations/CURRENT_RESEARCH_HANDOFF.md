# Current research handoff

| status | current truth |
|---|---|
| progress | `TASK-20260905-03BFIX` is implemented and run v2 completed: `108/108` corpus plus `20/20` screen clips passed separate-process full-clip replay and per-step plain-runtime comparison (`128,000/128,000` transitions); E3 qualified `28/36` blocks and `62/72` pairs. |
| bottleneck | The exact imported expert failed the frozen 20-reset development screen: `19/20` resets remained healthy and upright; median velocity `5.001603770686775 m/s` and displacement `20/20` passed. No retry or tuning occurred. |
| next step | Fable reviews and commits the builder slice, records the failed expert screen without widening the claim, preserves v1 as superseded, then folds any P1 review findings and proceeds to `03A2FIX2`. |

Fable resume: review the 03BFIX diff and validation manifest; commit the builder
slice and integration separately; record the failed screen in strategy and ADR
0005; preserve v1 as superseded; do not admit a tracker, E4, E5, oracle,
naturalness, or robustness claim; then fold the 03B reviews and launch
`03A2FIX2`.

- Updated: `2026-09-05T07:17:40Z`
- Repository: `/Users/samueldoane/Documents/ChatGPT/humanoid-harness`
- Branch: `main`
- 03BFIX clean-precondition HEAD: `a8314c9dfd3d6d09f1488935acf71c98f74180c3` (at or after required `a533606`; Slice 03B `41b2597` and integration already committed). Fable owns Git writes.
- Strategy control owner: Fable session `807bcdb2-462c-4ea8-803a-1e4b41259e12`.
- Current write lease: `CLAIMED` by `sol-builder-20260905-03bfix`, model
  `gpt-5.6-sol`, role `builder`, with the exact 03BFIX scope; the launcher owns
  renewal and release.
- Active write worker: `sol-builder-20260905-03bfix` (packet `TASK-20260905-03BFIX`). Read-only: scientific and robustness reviews of `41b2597`, packet B design survey `sol-survey-20260905-b`.
- Takeover authorization: disabled; the heartbeat may only report
- Human gates: all three startup gates answered; see "Questions for Samuel" below

## `TASK-20260905-03BFIX` completed run v2

- lease: exact launcher-owned lease remained `CLAIMED` by
  `sol-builder-20260905-03bfix`, model `gpt-5.6-sol`, role `builder`, and the
  packet's exact scope. The builder did not renew or release it.
- code repair: removed the live post-step `mj_forward`; policy input and
  canonical boundary observations now use copied `reset`/`step` returns; torso
  orientation comes from copied `qpos`; no collector, screen, or live-certifier
  path calls `mj_forward`, `mj_step1`, or `mj_kinematics`.
- comparison: all `108` corpus clips and `20` screen clips carry per-step
  bitwise checks for action bytes, integration state, explicit `qpos` and
  `qvel`, `cfrc_ext`, canonical and returned observations, reward, simulation
  time, wrapper counter and flags, and result flags.
- replay: method `predecessor_transition_cache_rebuild/v2`; for `t > 0`, restore
  boundary `t-1`, execute and discard action `t-1`, verify boundary `t`, then
  execute and compare the target transition. Boundary `0` is verified directly
  from the seeded reset.
- run command: `/usr/bin/time -p .venv/bin/python -m
  oracle_composition.experiments.reference_corpus_runner --run-version v2`;
  exit `0`; `291.64 s` real, `287.59 s` user, `2.84 s` sys. This total includes
  collection and the separate-process certifier; a certifier-only wall timer
  was not retained.
- reset-free actor preconditions:

  | actor | import receipt SHA-256 | strict NPZ SHA-256 | equivalence receipt SHA-256 |
  |---|---|---|---|
  | expert | `b790f06ccb66ca45809eaa1b0c5cd6804e072c6fd9eb48c61838ee2e558bd9d7` | `60987a4e054db2e04f9cb3ab73e13dfe8e2f3ec7dec46346d2b9d0277ad18d9b` | `65b2090b783e381e799e90e372308018d4e9a50fa6be2df7934af5f24e78a23e` |
  | medium | `b1c07c2f5070b48ff6bb28983b16ed16d1616e7ad18f557639dad89bb1f4f172` | `2677ebb70cd20e0ba8f8a591814fc853e325277db65184ebafb5bf5e21198b04` | `ede73be8db0ec3411dfb1a5ce2dbbe109d49432d1aeba97e6e4b9bf4fb44d487` |
  | simple | `bef2a2678d30f13a9e3350bf8c8f591661bb129b063276a15b8051f94ee313f7` | `b09aa921316640024e9703671d328d7ecbabe76f04cfed8c1d8fb86fbd95a917` | `604db637d316d3b8e46fb6f5d7a9b9004cfced267836394d89df183c87615268` |

- run-level receipt hashes:

  | artifact | raw SHA-256 | result |
  |---|---|---|
  | E1 initialization identity | `28725ecfba2ca89f4b4608e5e8d5b5024cfe2ed8df71383bc03a248891edf266` | pass |
  | frozen E3 manifest | `003b5b045ca9f8af2b0af3f559af206e1000835deea98ff98bd45a79fa5b9504` | unchanged |
  | sandbox baseline failures | `c06586e4449fad6b00e6356e6a70e2d75e3476fb8f4fbbf0b2c4e72c547a8345` | 44 IDs |
  | corpus manifest | `a3e9a7234b67194f0d4ac8d3961e9040f217ec9768e27451c279105aa3391090` | 108 corpus clips |
  | attempt ledger | `2600d408122abb5d2f2d981a75241ff7fc8dea7db2e6a50a5a0e61f4939bdf5d` | 128 one-shot attempts |
  | runner log | `2d37545f65482d8bec1566a983ed90ca8f01cc6d4814ed57018451c338423008` | exit 0 |
  | Tier-D certifier request | `cc2d70800c59828716950365da678f85c2b04422d797ee859e828672c94acac4` | 128 bundles, replay method v2 |
  | Tier-D aggregate | `dcf06c8e96ebd4d7c3171ab6d3062a73045addcb1be992e0e2d3ca5f62801bf7` | internal `d70183390664cff07730814c72cb34aec668b324060f0d4978a0175b6d24e584` |
  | E3 fork certificate | `5a91a265e057e6f8c6497d50a4a20ac3baab40b67c101115bc96b6f4b7555b74` | internal `eaeab5fb7a06b9419233b553c5fc70eca16eb62fe6d119b842c86291a306ae4d` |
  | expert screen receipt | `0ff4bd14712729ca2bdced8d5c8511317ab0f118a98adce6f75557e24eff5a60` | internal `ecd9f07077ddf29772008b8e8ff5676609e75fbd893c9a41b4d27e2dcd577c03` |
  | content index | `9347c741241b1138ce361f01ad319d05e09a7597c9ef51d9715b86dee6c34afa` | internal `c3cf17100248a05bc609d28ff4a7bb7fbc55b61ffb6d1d6a2a4e34b23b2b6ba8` |
  | run result | `c41d78ff0c1d2fdd899a53cf419b0b0d3f43b8d9d24e072cca400af42e6f1e98` | completed; screen false |
  | full-suite log | `ebff2d7543398816dcd6fec88858a9313f2ccb82ebe2f322ac24cd231b873f00` | exact baseline |
  | v2 validation manifest | `5eeedc84d13bf83fde534423fb5eb898160f543ef7dcdf325a7e136adbcfc434` | payload-free; all 128 per-clip receipt chains |

  The validation manifest records every clip's bundle core and manifest,
  payload, reference identity, plain-comparison object, boundary-receipt array,
  transition-receipt array, Tier-D certificate, and replay-verification digest.

- replay and plain-comparison counts:

  | clip group | actor | clips | Tier-D pass/fail | plain pass/fail | transitions |
  |---|---|---:|---:|---:|---:|
  | corpus | expert | 36 | 36 / 0 | 36 / 0 | 36,000 |
  | corpus | medium | 36 | 36 / 0 | 36 / 0 | 36,000 |
  | corpus | simple | 36 | 36 / 0 | 36 / 0 | 36,000 |
  | screen | expert | 20 | 20 / 0 | 20 / 0 | 20,000 |
  | total |  | 128 | 128 / 0 | 128 / 0 | 128,000 |

- E3 results (`M` is expert-vs-medium, `S` expert-vs-simple):

  | seed | M | S | block |
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

  Summary: blocks `28` pass / `8` fail; medium pairs `34/36`; simple pairs
  `28/36`; qualifying pairs `62/72`. Branches: expert `34/36`, medium `36/36`,
  simple `29/36`. All ten failed pairs came only from the locked branch
  horizon/collapse/contact/replay conjunction; action and future-variation
  thresholds passed.
- screen result: **failed**. Healthy `19/20` (required `20`), upright `19/20`
  (required `20`), median forward velocity `5.001603770686775 m/s` (required
  at least `0.5`), displacement at least `5 m` in `20/20` episodes (required
  at least `18`). Seed `96018` first became unhealthy at step `860` and first
  not upright at step `871`; minimum height `0.16328118194073535 m`, upright
  fraction `0.87`, displacement `65.41267369164503 m`, velocity
  `4.360844912776202 m/s`, non-foot contact fraction `0.107`. No replacement,
  retry, or threshold change.
- tests and checks: focused corpus suite `33 passed in 1.93 s`; final focused
  rerun `33 passed in 1.87 s`; ordered corpus
  plus TQC-runtime suite `34 passed, 9 failed in 2.18 s`, with only the expected
  `cpu_model=arm` failures; full suite with the exact reward-lane test deselected
  `1,254 passed, 44 failed, 11 skipped, 1 deselected in 117.11 s` (`117.54 s`
  process wall), with failure IDs exactly equal the saved baseline; Ruff lint
  passed; Ruff format reports `231` files already formatted; `git diff --check`
  passed.
- v1 preservation: all named v1 artifact hashes and the v1 validation hash
  `aa17751cb665afef0cd4d8f9a353e7889bb227f3819de98c5de9a92db48a3646`
  remain unchanged. V1 is an instrumented-schedule run superseded by v2, not a
  plain-runtime result.
- claim ceiling: only the named clips as plain-runtime closed-loop trajectories
  of the exact imported actors with full-clip replay certificates and per-step
  plain-comparison receipts; a qualifying E3 fork corpus; and the exact expert's
  failed predeclared screen in the pinned local runtime. No tracker, E4, E5,
  oracle, naturalness, or robustness claim.
- final audit at `2026-09-05T07:17:40Z`: all `128` bundle, payload,
  plain-comparison, Tier-D, and replay-verification bindings; all `83`
  content-addressed objects; and the seven named immutable-v1 hashes passed.
  Validation SHA-256 remains
  `5eeedc84d13bf83fde534423fb5eb898160f543ef7dcdf325a7e136adbcfc434`.

`TASK-20260904-03B` checkpoint changed-files list:

- `src/oracle_composition/contracts/reference_identity_v2.py`
- `src/oracle_composition/envs/reference_corpus.py`
- `src/oracle_composition/sources/strict_tqc_actor_runtime.py`
- `src/oracle_composition/experiments/reference_corpus_contract.py`
- `src/oracle_composition/experiments/reference_corpus_collector.py`
- `src/oracle_composition/experiments/reference_corpus_bundle.py`
- `src/oracle_composition/experiments/reference_corpus_certifier.py`
- `src/oracle_composition/experiments/e3_fork_certifier.py`
- `src/oracle_composition/experiments/expert_development_screen.py`
- `src/oracle_composition/experiments/reference_corpus_runner.py`
- `tests/experiments/test_reference_corpus_v1.py`
- `experiments/reference_corpus_v1/e3_manifest_v1.json`
- `artifacts/reference_corpus_v1/sandbox_baseline_failures.txt` (ignored local receipt)
- `docs/operations/CURRENT_RESEARCH_HANDOFF.md`

`TASK-20260905-03BRUN` stop receipt:

- preconditions: exact lease `CLAIMED`; only the authorized 03B dirty paths;
  three strict NPZ hashes matched `PUBLIC_EXPERT_IMPORT.md`; E3 manifest SHA-256
  `003b5b045ca9f8af2b0af3f559af206e1000835deea98ff98bd45a79fa5b9504`
- repair: removed `torch.set_num_threads` and `torch.set_num_interop_threads`
  from the in-process actor runtime; added a test binding intra-op, inter-op, and
  deterministic-algorithm state before and after actor construction
- tests: `tests/experiments/test_reference_corpus_v1.py` -> `28 passed`; ordered
  corpus plus TQC-runtime files -> `29 passed, 9 failed` only on the saved
  sandbox `cpu_model = arm` mismatch; the runtime file alone -> `1 passed, 9
  failed` on the identical IDs and cause; no thread-field failure and no error
- lint: focused `ruff check` passed; repository `ruff format --check` reported
  all `230` files formatted; `git diff --check` passed
- command: `.venv/bin/python -m
  oracle_composition.experiments.reference_corpus_runner`
- stop: exit `1` after `0.48 s`, before any reset, at
  `_load_json(external_actor_import_expert_03a3_v1.json)` with `JSON artifact is
  not canonical`; the pinned raw hash is
  `b790f06ccb66ca45809eaa1b0c5cd6804e072c6fd9eb48c61838ee2e558bd9d7`
- diagnosis: each of the three import and three equivalence receipts is exactly
  `canonical_json_bytes(parsed_value) + b"\n"`; no payload or receipt was
  changed, and no retry was made

`TASK-20260905-03BRUN2` stopped-run receipt:

- loader repair: `_load_json(..., require_terminating_lf=True)` now admits only
  canonical JSON plus one LF for the six importer-owned receipts. The regression
  accepts one LF and rejects zero or two. Importers and receipt bytes are
  unchanged.
- reset-free preflight:

  | actor | import receipt SHA-256 | strict NPZ SHA-256 | equivalence receipt SHA-256 |
  |---|---|---|---|
  | expert | `b790f06ccb66ca45809eaa1b0c5cd6804e072c6fd9eb48c61838ee2e558bd9d7` | `60987a4e054db2e04f9cb3ab73e13dfe8e2f3ec7dec46346d2b9d0277ad18d9b` | `65b2090b783e381e799e90e372308018d4e9a50fa6be2df7934af5f24e78a23e` |
  | medium | `b1c07c2f5070b48ff6bb28983b16ed16d1616e7ad18f557639dad89bb1f4f172` | `2677ebb70cd20e0ba8f8a591814fc853e325277db65184ebafb5bf5e21198b04` | `ede73be8db0ec3411dfb1a5ce2dbbe109d49432d1aeba97e6e4b9bf4fb44d487` |
  | simple | `bef2a2678d30f13a9e3350bf8c8f591661bb129b063276a15b8051f94ee313f7` | `b09aa921316640024e9703671d328d7ecbabe76f04cfed8c1d8fb86fbd95a917` | `604db637d316d3b8e46fb6f5d7a9b9004cfced267836394d89df183c87615268` |

- run receipts:

  | artifact | SHA-256 | result |
  |---|---|---|
  | E3 frozen manifest | `003b5b045ca9f8af2b0af3f559af206e1000835deea98ff98bd45a79fa5b9504` | precondition pass |
  | E1 initialization-identity receipt | `28725ecfba2ca89f4b4608e5e8d5b5024cfe2ed8df71383bc03a248891edf266` | precondition pass |
  | sandbox baseline failure IDs | `c06586e4449fad6b00e6356e6a70e2d75e3476fb8f4fbbf0b2c4e72c547a8345` | `44` expected IDs |
  | corpus manifest | `87c25d9b380605556a62681e44879943275a358b7136b322e0b103f2b8dff9f5` | `108` clips |
  | attempt ledger | `29c03cbbd89cbe3045021e54bd2edb6e0a47f8624d152aefa22f01377e24c5e6` | stopped attempt retained |
  | runner log | `03d85f84bb6ebb06c6fe15ec4aa4fb9262895d3525158f482bb45844a680f6f2` | exit `1`, `61.57 s` |
  | certifier request | `05b22aadc791b873d7e3d8aa17558b154d200603d13e5341a2174b9a67f0af51` | `108` bundles |
  | certifier log | `30c87b8977a0cc0aca1a8c6cf3ed2d0544491fd92befe9c0f2d7ac9d3248b4c9` | exit `0`, `64.26 s` |
  | stopped-run Tier-D aggregate | `3b80d2b3afa5a3206004acd5cc3507cfc4483e3feac6254545ce8a395fc4d616` | binds all `108` per-clip certificates |
  | E3 certificate | `c977f5068e7c71bcd75a91da3ccf5fc5825e63f3cad4d1608f4106ba64be50fa` | qualifying corpus |
  | stopped-run content index | `7d9bc501f77466f7dc77151812c8024e6050e6ceda985a9e286a49628387acbb` | partial evidence only |
  | payload-free validation manifest | `aa17751cb665afef0cd4d8f9a353e7889bb227f3819de98c5de9a92db48a3646` | binds all `108` certificate hashes |
  | full-suite log | `6787a1c8f50cf1ef7f5100a2e8082faadff4e25a24312f0fe206e75aca777fa8` | exact sandbox baseline |

- Tier-D per actor: expert `36` pass / `0` fail; medium `36` pass /
  `0` fail; simple `36` pass / `0` fail. Every clip verified all `1,000`
  transitions, for `108,000/108,000` transitions total.
- E3 results (`M` is expert-vs-medium, `S` expert-vs-simple):

  | seed | M | S | block |
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

  Summary: blocks `27` pass / `9` fail; medium pairs `32/36`; simple
  pairs `29/36`; qualifying pairs `61/72`. Branches: expert `34/36`, medium
  `34/36`, simple `31/36`. All pair failures came from the locked branch
  horizon/collapse/contact/replay conjunction, not action or future-variation
  thresholds.
- screen result: seed `96001`, reset order `108`, raised
  `ReferenceCorpusContractError: plain/instrumented reward canary differs`;
  `0/20` clips completed; the four locomotion gates were not computed; screen
  pass/fail is **no result**; screen receipt is absent. No retry occurred.
- leading diagnosis, not a measured fix: Gymnasium reset already calls
  `mj_forward`; `_capture_boundary` then calls it again only on the instrumented
  environment before the first action. The combined guard does not retain which
  of returned observation, reward, or flags differed. Preserve the hard gate.
- validation: focused corpus suite `29 passed`; repository suite `1,250 passed,
  44 failed, 11 skipped, 1 deselected` in `114.24 s`, with all `44` failure IDs
  exactly equal to `sandbox_baseline_failures.txt`; the reward-lane replay test
  was deselected by exact name and its receipt was untouched; Ruff lint, Ruff
  format (`231` files), and `git diff --check` pass.
- changed tracked paths in this continuation: runner, corpus test, root README,
  experiment registry, new `experiments/reference_corpus_v1/PROTOCOL.md`, new
  payload-free validation manifest, and this handoff. The pre-existing
  reward-lane receipt remains outside this worker's changes.
- final audit at `2026-09-05T06:03Z`: the exact launcher-owned lease remained
  `CLAIMED`; validation target bindings and all `108` certificate entries
  revalidated; only the authorized 03B/documentation paths and the pre-existing
  reward-lane receipt were dirty. The builder did not renew, release, or write
  Git state.

## Active read-only Sol workers

| owner | role | packet | run directory | screen | runner PID |
|---|---|---|---|---|---|
| `sol-review-sci-20260904-01` | scientific reviewer | `.orchestration/task-packets/TASK-20260904-01-strategy-scientific-review.md` | `.orchestration/sol-runs/20260904T162138Z-91fdd351-b579-46f8-b317-402f78bd2907` | `hh-sol-5538c277-e542-4505-8c92-787b4798b0d8` | `33469` |
| `sol-review-adv-20260904-02` | adversarial reviewer | `.orchestration/task-packets/TASK-20260904-02-strategy-adversarial-review.md` | `.orchestration/sol-runs/20260904T162140Z-53837dc0-a9f8-4bf5-97f7-ae7e593a60b4` | `hh-sol-f4c09dad-bbe5-40f5-8587-99155b3b88d9` | `33916` |
| `sol-design-reward-20260904-04` | research designer | `.orchestration/task-packets/TASK-20260904-04-reward-track-design-survey.md` | `.orchestration/sol-runs/20260904T163715Z-ddddde81-0085-49fd-8714-0d512c9d39a0` | `hh-sol-342b73b1-463f-4fd0-9f2d-618598681f8f` | `37865` |
| `sol-builder-20260904-03a` (write) | builder | `TASK-20260904-03A` | `.orchestration/sol-runs/20260904T164729Z-273705b5-fd1b-49d6-9495-98e63f907b01` | stopped at the `.git` boundary; superseded | `50887` |
| `sol-builder-20260904-03a2` (write, launch 1) | builder | `TASK-20260904-03A2` | `.orchestration/sol-runs/20260904T165851Z-cde09967-82e9-425c-9b16-91292ac226f5` | stopped: two-dot diff precondition | `58734` |
| `sol-builder-20260904-03a2` (write, launch 2) | builder | `TASK-20260904-03A2` | `.orchestration/sol-runs/20260904T170134Z-f48b6fc2-905b-4323-87f3-8b88246888c4` | stopped: in-sandbox suite failures | `59876` |
| `sol-builder-20260904-03a2` (write, launch 3) | builder | `TASK-20260904-03A2` | `.orchestration/sol-runs/20260904T171428Z-2d37493e-5ef0-455c-8c58-5f9878a2074b` | partial slice in the working tree; stopped at `process safe globals refused` | `62996` |
| `sol-builder-20260904-03a2r` (write) | builder | `TASK-20260904-03A2R` | `.orchestration/sol-runs/20260904T174545Z-7b364f7f-b3bf-426d-8023-17389e1f0c12` | `SUCCEEDED`: slice complete, thread `01a06d86-cd17-7ee2-b6de-c7d3d07576f9` | `67383` |
| `sol-review-sci-20260904-06` | scientific reviewer of `1d6b461` | `TASK-20260904-06` | `.orchestration/sol-runs/20260904T181613Z-721ea785-8dc3-4d8f-9eb6-84b4672d1fb3` | `hh-sol-869272fb-cc6c-480d-9b0a-0b3eece424a9` | `74477` |
| `sol-review-adv-20260904-07` | adversarial reviewer of `1d6b461` | `TASK-20260904-07` | `.orchestration/sol-runs/20260904T181614Z-5bb41e36-fc24-49b0-8fb3-a73409210b5a` | `hh-sol-d7eb4fef-93a3-49a5-9161-140a7d1672a7` | `74544` |
| `sol-builder-20260904-b0` (write) | builder | `TASK-20260904-B0` | `.orchestration/sol-runs/20260904T181636Z-8fcc968b-0d9a-4901-ba7f-3a389fc522af` | `SUCCEEDED`; slice uncommitted pending B0FIX; thread `01a06da3-08f9-7a92-b01e-eb2bacdb3502` | `76786` |
| `sol-builder-20260904-b0fix` (write) | builder | `TASK-20260904-B0FIX` | `.orchestration/sol-runs/20260904T202141Z-260b15ed-0b8b-47df-a17b-800bc2267504` | `SUCCEEDED`; thread `01a06e15-8f53-7452-ad7a-00a9d3688a14` | `46776` |
| `sol-review-adv-20260904-08` | robustness reviewer of `1d6b461` | `TASK-REVIEW-ROBUSTNESS-GENERIC` | `.orchestration/sol-runs/20260904T202141Z-fe0d2655-8bd5-4363-8445-abf0e2d51bda` | `SUCCEEDED`; `ACCEPT-WITH-REPAIRS`; thread `01a06e15-8faa-7453-8319-d724dc56ff5f` | `46844` |
| `sol-builder-20260904-b0fix2` (write) | builder | `TASK-20260904-B0FIX2` | `.orchestration/sol-runs/20260904T210814Z-f73f6df5-757f-4fab-b3aa-17f8c660935b` | `SUCCEEDED`; thread `01a06e40-2e02-7540-843e-b7b908f1ae37` | `60764` |
| `sol-builder-20260904-b0fix3` (write) | builder | `TASK-20260904-B0FIX3` | `.orchestration/sol-runs/20260904T214238Z-b4e1b91c-2e42-434c-b3e7-776536d48e50` | `SUCCEEDED`; thread `01a06e5f-ad2e-7852-a327-f8342a75cc73` | `73489` |
| `sol-builder-20260904-b0fix4` (write) | builder | `TASK-20260904-B0FIX4` | `.orchestration/sol-runs/20260904T215855Z-02389704-6e96-4274-9b41-2b1a06b5f8ad` | `SUCCEEDED`; thread `01a06e6e-9bf7-79f3-a373-b5ed704cd40b` | `78268` |
| `sol-design-tracker-20260904-05` | research designer | `.orchestration/task-packets/TASK-20260904-05-tracker-track-design-survey.md` | `.orchestration/sol-runs/20260904T164217Z-62e441c4-8938-4986-b267-cb0f8d21ce80` | `hh-sol-25fc9414-83ae-42f7-929f-a3f3c2e5d37d` | `47373` |

Launched `2026-09-04 16:21Z`. Poll with `./scripts/start-sol-worker status RUN_DIR`;
read only `final.txt` and `result.json`, never stream `events.jsonl`. Codex
thread IDs observed at `16:25Z`: scientific `01a06d39-cb22-7af2-a89f-88c80845f1d3`,
adversarial `01a06d39-d229-7b12-81ec-cad50a5a3a90`.

State at `2026-09-04T16:46Z`: reviews `01` and `02` are terminal `SUCCEEDED`, both
`GO-WITH-FIXES` with no P0 (scientific `4` P1, adversarial `9` P1 and `1` P2);
all P1 fixes are folded into ADR 0005, the strategy, and the split builder
packets. Survey `04` running since `16:37Z`; survey `05` running since
`16:42Z`. The original packet `TASK-20260904-03` is superseded by `03A` (this
launch) and a later `03B` (external evaluator adapter, 20-reset screen, replay
bundle).

## Questions for Samuel

| id | question | answer line to add here |
|---|---|---|
| Q1 | May the builder import the public Farama `Humanoid-v5` TQC expert through the data-only path in ADR 0005 (pinned SHA-256 bytes, `torch.load(weights_only=True)` tensors only, JSON-only metadata, no `TQC.load`)? The E1 note currently forbids loading uploaded SB3 or Torch objects. If no: authorize up to `3` receipted local TQC attempts of at most `3 h` CPU each. | `external-checkpoint-import: approved YYYY-MM-DD` or `external-checkpoint-import: denied; local-attempt-budget: N` |
| Q2 | May a reward-only harness loop on stock `Humanoid-v5` (route D in the strategy) run as a parallel second track while the tracker is admitted? It reorders `LG-11`. | `reward-first-track: approved` or `denied` |
| veto | The builder will move the uncommitted TQC-v2 files byte-exact to branch `wip/tqc-v2-attempt-supervisor` and remove them from the `main` working tree. | `wip-branch-move: vetoed` stops it |

Answers recorded `2026-09-04T16:31Z` from Samuel's message in the Fable session:

- `external-checkpoint-import: approved 2026-09-04`
- `reward-first-track: approved`
- `wip-branch-move: approved`
- Samuel ran `/status` and confirmed `claude-fable-5-1`; the identity gate is
  human-confirmed for this session.
- Standing instruction: do not ask about routine route decisions or worker
  launches; run Sol workers directly, delegate token-heavy work to them, and
  keep running autonomously. The contract's hard gates (real 1M-step attempt,
  PRAXIST campaign, formal confirmatory study, paid API usage, push or
  publication) still need explicit authorization.

## Claim boundary

| layer | current statement |
|---|---|
| research target | An LLM-guided harness revises a reference-composition oracle `O_k` and task reward `r_k`, using protected rollout evidence and optional human steering. |
| implemented capability | Typed oracle artifacts; a validated linear phase-window automaton; strict public-actor NPZ runtime; observation-preserving collection with per-step plain comparison; complete integration-state/reference/replay bundle contracts; predecessor-transition cache reconstruction in a separate-process certifier; E3 fork certifier; Gymnasium adapter; deterministic trace/metric contracts; CLI; read-only evidence UI; research-source ledger. |
| measured evidence | Interface and regression checks; one non-admitted falling tracker exploration; one reviewed offline numeric-reference sensitivity probe; one 100k resource calibration; three hash-pinned public actors; run-v2 `108/108` corpus and `20/20` screen clips with full replay and plain-comparison receipts; `28/36` qualifying E3 blocks; exact expert screen failure at `19/20` healthy and upright resets while velocity and displacement passed. |
| not demonstrated | Expert development-screen pass; stable reference tracking; causal policy use of reference windows; better transitions; recovery; oracle improvement; reward improvement; general task composition; naturalness; robustness; cross-MDP generalization; an autonomous closed research loop. |

## Scientific dependency chain

```text
Experiment 001             Experiment 002                 Experiment 003+
admit stable tracker  -->  prove causal reference use --> compare oracle arms
   ROUTE: ADR 0005              BLOCKED                   NOT AUTHORIZED
```

- A time-indexed reward is not proof that actor and critic consumed the
  immutable reference clock and window.
- Oracle quality cannot be inferred from the current falling video.
- Reference generation and reference composition are separate problems. This
  project currently owns composition.
- Task-reward generation is the approved parallel Family-B track (ADR 0006).
  It stays fixed inside every oracle-only comparison.
- The public expert generated the registered Minari expert clips. A tracker
  built on it can follow its own rollout while ignoring the reference; gate E3
  and non-self reference gaits are mandatory.

## Route decision

Recorded in `docs/strategy/RESEARCH_STRATEGY.md`, section "2026-09-04 startup
audit", and `docs/decisions/0005_public_expert_base_controller.md`.

| route | verdict |
|---|---|
| Finish the TQC-v2 one-attempt supervisor | parked: `8` open P1 lifecycle findings and `4,499` uncommitted lines guarding a `27`-minute job |
| Public expert import, data-only, receipted local fallback | recommended; scientific review `GO-WITH-FIXES` folded; adversarial review pending |
| Reward-first loop on stock `Humanoid-v5` | approved parallel track (ADR 0006); design survey running |

## Working-tree state

- Current 03BFIX state: uncommitted changes are limited to the ten authorized
  source/test files, `README.md`, `experiments/README.md`, this handoff,
  `experiments/reference_corpus_v1/PROTOCOL.md`, and the new payload-free
  `experiments/reference_corpus_v1/reviews/corpus_run_v2_validation.json`.
  Run outputs are ignored under `artifacts/reference_corpus_v2/`. No Git write
  command was run, and no v1 artifact or v1 validation file changed.
- The following older entries remain as historical handoff context.
- Start state was clean at `main` HEAD
  `48955dfb299498d1893e6dfffe9facb87a4192a5`; the three-dot comparison and
  commit `5bdae45` each list exactly the expected 20 preserved paths.
- `TASK-20260904-03A3` changed-files list and receipts:
  - Preconditions passed at `main` HEAD
    `425d3bcf203929403205a8e8f01c0cc453b60b0b`: the working tree was clean,
    the exact `sol-builder-20260904-03a3` lease was `CLAIMED`, and the 03A2
    expert chain was present. The required pre-change suite was `44 failed,
    1198 passed, 11 skipped`; the sorted baseline is
    `artifacts/bootstrap_tqc_humanoid/sandbox_baseline_failures_03a3.txt`,
    SHA-256
    `c06586e4449fad6b00e6356e6a70e2d75e3476fb8f4fbbf0b2c4e72c547a8345`.
  - All 18 present source files across the `medium` and `simple` local copies
    independently matched their API byte counts and `lfs.sha256` or computed
    Git-blob SHA-1 before any load. Both policy files also matched their pinned
    SHA-256 and byte counts.
  - Trackable files:
    - `research/source_controllers/farama_minari_humanoid_v5_tqc_medium/README.md`.
    - `research/source_controllers/farama_minari_humanoid_v5_tqc_medium/RECEIPT.json`.
    - `research/source_controllers/farama_minari_humanoid_v5_tqc_simple/README.md`.
    - `research/source_controllers/farama_minari_humanoid_v5_tqc_simple/RECEIPT.json`.
    - `src/oracle_composition/sources/farama_tqc_sibling_registrations.py`.
    - `src/oracle_composition/sources/external_sb3_actor.py`.
    - `src/oracle_composition/sources/external_payload_policy.py`.
    - `src/oracle_composition/experiments/external_tqc_actor_equivalence.py`.
    - `src/oracle_composition/experiments/external_tqc_initialization_identity.py`.
    - `tests/sources/test_farama_tqc_sibling_registrations.py`.
    - `tests/sources/test_external_sb3_actor_siblings.py`.
    - `tests/sources/test_external_sb3_actor.py`.
    - `tests/experiments/test_external_tqc_initialization_identity.py`.
    - `tests/policy/test_no_external_payload_in_index.py`.
    - `experiments/bootstrap_tqc_humanoid/PUBLIC_EXPERT_IMPORT.md`.
    - `experiments/bootstrap_tqc_humanoid/E1_INITIALIZATION_IDENTITY.md`.
    - `docs/operations/CURRENT_RESEARCH_HANDOFF.md`.
  - Source registration receipts:

    | variant | repository commit | receipt SHA-256 | bytes |
    |---|---|---|---:|
    | `medium` | `949f7963c1a8964587dca73d48873ad021e168b5` | `68344e980e42543ddd5ca3d164a9c8173f63950d47b4c8c9064e57283fd4e0a1` | 9,401 |
    | `simple` | `39e2954c193fc1352535935a7d71eca9e5745d9b` | `745d82dea8fc4878f70ed58f4d6ff4cd506b0f94c85676dc70e3cffce8283ac5` | 9,401 |

  - Local ignored import artifacts:

    | variant | import receipt SHA-256 | strict NPZ SHA-256 | equivalence receipt SHA-256 |
    |---|---|---|---|
    | `expert` | `b790f06ccb66ca45809eaa1b0c5cd6804e072c6fd9eb48c61838ee2e558bd9d7` | `60987a4e054db2e04f9cb3ab73e13dfe8e2f3ec7dec46346d2b9d0277ad18d9b` | `65b2090b783e381e799e90e372308018d4e9a50fa6be2df7934af5f24e78a23e` |
    | `medium` | `b1c07c2f5070b48ff6bb28983b16ed16d1616e7ad18f557639dad89bb1f4f172` | `2677ebb70cd20e0ba8f8a591814fc853e325277db65184ebafb5bf5e21198b04` | `ede73be8db0ec3411dfb1a5ce2dbbe109d49432d1aeba97e6e4b9bf4fb44d487` |
    | `simple` | `bef2a2678d30f13a9e3350bf8c8f591661bb129b063276a15b8051f94ee313f7` | `b09aa921316640024e9703671d328d7ecbabe76f04cfed8c1d8fb86fbd95a917` | `604db637d316d3b8e46fb6f5d7a9b9004cfced267836394d89df183c87615268` |

  - Actual E1 ignored receipt
    `artifacts/bootstrap_tqc_humanoid/e1_initialization_identity_external_v1.json`:
    SHA-256
    `28725ecfba2ca89f4b4608e5e8d5b5024cfe2ed8df71383bc03a248891edf266`,
    11,023 bytes. Both contenders pass bitwise identity on the imported expert;
    authority is `external_pretrained_artifact`, training/environment steps are
    zero, and behavior was not evaluated.
  - Verification: focused import, equivalence, E1, payload-policy, and retained
    fixture tests `166 passed`; source/API/registration/import/NPZ/equivalence/E1
    revalidation passes; Ruff lint and format checks pass for 219 files;
    `git diff --check` passes. Post-change full suite `45 failed, 1221 passed,
    11 skipped`; the 44 baseline IDs remain exact and one additional failure is
    `tests/experiments/test_reward_target_speed_manifest.py::test_recorded_no_learning_runtime_receipt_replays_exactly`.
    It is solely the protected Family-B source-tree binding (`7496476d...319c9c`
    recorded versus `ba4faf45...23a13d` observed).
  - Stop boundary: the required post-change baseline gate is not met. The
    Family-B receipt is outside this scope and forbidden, so the builder did
    not refresh it or weaken its validator.
  - Fable resume: review the 03A3 diff and receipts, refresh the Family-B
    no-learning runtime receipt under its owning authority, rerun the full
    suite, and commit only after the additional failure clears.
- `TASK-20260904-03A2FIX` changed-files list and receipts:
  - `src/oracle_composition/experiments/external_tqc_actor_equivalence.py`.
  - `src/oracle_composition/sources/_external_sb3_actor_worker.py`.
  - `src/oracle_composition/sources/external_payload_policy.py`.
  - `src/oracle_composition/sources/external_sb3_actor.py`.
  - `src/oracle_composition/sources/farama_tqc_registration.py`.
  - `tests/policy/test_no_external_payload_in_index.py`.
  - `tests/sources/test_external_sb3_actor.py`.
  - `tests/sources/test_farama_tqc_registration.py`.
  - `experiments/bootstrap_tqc_humanoid/PUBLIC_EXPERT_IMPORT.md`.
  - `docs/operations/CURRENT_RESEARCH_HANDOFF.md`.
  - Regenerated ignored import receipt
    `artifacts/bootstrap_tqc_humanoid/external_actor_import_v1.json`: schema
    v2, SHA-256
    `c2379a21079c40928548b593cb86a8fd8eea468787cf31c28481792d3121e80b`,
    12,465 bytes.
  - Regenerated ignored equivalence receipt
    `artifacts/bootstrap_tqc_humanoid/external_actor_equivalence_v1.json`:
    schema v2, SHA-256
    `0d5ec467051e2eebc16a0a186afbe85cc1364afbeab8fcbefa41f088cb656f31`,
    2,687 bytes.
  - Unchanged registration receipt
    `research/source_controllers/farama_minari_humanoid_v5_tqc_expert/RECEIPT.json`:
    SHA-256
    `5ba0845e8b0cd9b6f39c956ddc46d0f46e8e08690d8bdf832941a1f914a8e0cf`,
    9,395 bytes.
  - Unchanged actor NPZ
    `artifacts/bootstrap_tqc_humanoid/farama_minari_humanoid_v5_tqc_actor_v1.npz`:
    SHA-256
    `60987a4e054db2e04f9cb3ab73e13dfe8e2f3ec7dec46346d2b9d0277ad18d9b`,
    618,674 bytes; unchanged actor-state fingerprint
    `3fd39cc715a10126fd92b20f6ce213c380eb4d5df843a42315aac50cf116748a`.
  - Reused sandbox baseline receipt
    `artifacts/bootstrap_tqc_humanoid/sandbox_baseline_failures.txt`: SHA-256
    `c06586e4449fad6b00e6356e6a70e2d75e3476fb8f4fbbf0b2c4e72c547a8345`.
  - Verification: focused source/policy/equivalence tests `173
    passed, 1 skipped`; the skip is `MPS is unavailable`. Full suite `45
    failed, 1197 passed, 11 skipped`: all 44 saved sandbox failures plus only
    `tests/experiments/test_reward_target_speed_manifest.py::test_recorded_no_learning_runtime_receipt_replays_exactly`.
    The mismatch is confined to `source_tree_sha256`; the forbidden reward
    receipt was regenerated by Fable as an integration change (see the provenance correction below). Ruff lint passed and all 212 Python files are
    formatted. No training or behavior evaluation ran.
- `TASK-20260904-B0` tracked or trackable changed-files list:
  - `docs/operations/CURRENT_RESEARCH_HANDOFF.md`.
  - `experiments/family_b_target_speed_v1/PROTOCOL.md`.
  - `experiments/family_b_target_speed_v1/DECISION_RULE.md`.
  - `experiments/family_b_target_speed_v1/configs/family_b_target_speed_v1.study.json`.
  - `experiments/family_b_target_speed_v1/candidates/stock_r0.py`.
  - `experiments/family_b_target_speed_v1/candidates/manual_target_speed_v1.py`.
  - `experiments/family_b_target_speed_v1/receipts/builder_sandbox_canaries.json`.
  - `experiments/family_b_target_speed_v1/receipts/builder_synthetic_scale.json`.
  - `experiments/family_b_target_speed_v1/receipts/builder_runtime_no_learning_smoke.json`.
  - `src/oracle_composition/rewards/__init__.py`.
  - `src/oracle_composition/rewards/contract.py`.
  - `src/oracle_composition/rewards/stock_humanoid.py`.
  - `src/oracle_composition/rewards/static_validation.py`.
  - `src/oracle_composition/rewards/scale_calibration.py`.
  - `src/oracle_composition/rewards/sandbox.py`.
  - `src/oracle_composition/rewards/_sandbox_worker.py`.
  - `src/oracle_composition/experiments/reward_target_speed_evaluator.py`.
  - `src/oracle_composition/experiments/reward_target_speed_manifest.py`.
  - `tests/rewards/test_contract.py`.
  - `tests/rewards/test_stock_humanoid.py`.
  - `tests/rewards/test_static_validation.py`.
  - `tests/rewards/test_scale_calibration.py`.
  - `tests/rewards/test_sandbox.py`.
  - `tests/experiments/test_reward_target_speed_evaluator.py`.
  - `tests/experiments/test_reward_target_speed_manifest.py`.
  - `artifacts/family_b_target_speed_v1/sandbox_baseline_failures.txt` (ignored
    in-sandbox baseline receipt).
- `TASK-20260904-B0FIX` repair changed-files subset and receipts:
  - `src/oracle_composition/rewards/sandbox.py`.
  - `src/oracle_composition/rewards/_sandbox_worker.py`.
  - `tests/rewards/test_sandbox.py`.
  - `experiments/family_b_target_speed_v1/PROTOCOL.md`.
  - `experiments/family_b_target_speed_v1/receipts/builder_sandbox_canaries.json`:
    canonical SHA-256
    `b0cbe91ac54199906c4e68a785bb8af03cc0d9c904997a1a16845fdfa8eccacc`;
    file SHA-256
    `eb340a34e22e150506251e7d6322229b30d84f06cdf355dbd97ad92fa348d891`.
  - `experiments/family_b_target_speed_v1/receipts/host_sandbox_canaries.json`
    (unchanged historical diagnostic): canonical SHA-256
    `62c2529fd0430b0d9bd9fe7c0920015535961aa9771874aa65f0cecfad161190`;
    file SHA-256
    `1f29d913baabc1bcab49c1ed27417477c8950db3e8dba5a6bda77a962e9ad05d`.
  - `experiments/family_b_target_speed_v1/receipts/builder_runtime_no_learning_smoke.json`:
    canonical SHA-256
    `0f14245b8c63168e9f7d3d760a15bc2d21846abeeb2858c9e3bcaeaa9effab7c`;
    file SHA-256
    `8f216c6584610ef127909daba4309d38c6a53223114f0398d37fdf29c7203d0a`.
  - `docs/operations/CURRENT_RESEARCH_HANDOFF.md`.
  - Verification: `87 passed, 7 skipped` focused; full suite `44 failed, 1140
    passed, 9 skipped`, with the exact 44 IDs in
    `artifacts/family_b_target_speed_v1/sandbox_baseline_failures.txt`; Ruff
    lint and format checks pass.
- `TASK-20260904-B0FIX2` repair changed-files subset and receipts:
  - `src/oracle_composition/rewards/sandbox.py`.
  - `src/oracle_composition/rewards/_sandbox_worker.py`.
  - `tests/rewards/test_sandbox.py`.
  - `experiments/family_b_target_speed_v1/PROTOCOL.md`.
  - `experiments/family_b_target_speed_v1/receipts/builder_sandbox_canaries.json`:
    canonical SHA-256
    `22aa4e86fcafc52067bee2c8b34ec539b4f2ea2ba768807c12e8fb59b936bac6`;
    file SHA-256
    `e9349a74bdc933bef9a48f00fca3ad44dff342693246f943f7597e11a740fff6`;
    profile SHA-256
    `7dcef9b4e429752695700817d0748560f6832e2d8b6b1f6f4ec253f2a423a816`.
  - Interpreter launch path `.venv/bin/python` and resolved uv binary each
    hash to
    `7710b0490e6af648676d7ad163fa1d54bffb75d8505162609a1c3c9b76e1929d`.
  - `experiments/family_b_target_speed_v1/receipts/builder_runtime_no_learning_smoke.json`
    was refreshed only for source identity: canonical SHA-256
    `0ef5ce4bc3205c0b4d6abf84f60d9e0ab6028d265a28cca8f8c17cdb6a283063`;
    file SHA-256
    `164bdddf57a0f33e43cb57a391223190ef70a42b09c8f0312f008838bedc9d78`.
  - `experiments/family_b_target_speed_v1/receipts/host_sandbox_canaries.json`
    is unchanged historical profile-v1 evidence: canonical SHA-256
    `94b576cf6372902612e7cf68c2f9e5fbaf26bec707d5ebd52dc92aa79a68c786`;
    file SHA-256
    `a17291848ed15f210294f65849588456aa565bccde7715c39d5d160b0b9aeb4d`.
  - `docs/operations/CURRENT_RESEARCH_HANDOFF.md`.
  - Verification: sandbox tests `22 passed, 7 skipped` with exact reason
    `sandbox-exec: sandbox_apply: Operation not permitted`; full suite `44
    failed, 1143 passed, 9 skipped`, with the exact saved 44-test baseline;
    Ruff lint and format checks pass. No training or behavioral evaluation ran.
- `TASK-20260904-B0FIX3` repair changed-files subset and receipts:
  - `src/oracle_composition/rewards/sandbox.py`.
  - `tests/rewards/test_sandbox.py`.
  - `experiments/family_b_target_speed_v1/PROTOCOL.md`.
  - `experiments/family_b_target_speed_v1/receipts/builder_sandbox_canaries.json`:
    canonical SHA-256
    `3cfc64c0868c606cce9b400cf88493f2b9057d7c7dfe668cfd016b6114bf6238`;
    file SHA-256
    `ba1cb6246298d90f7011ed559c1d1829e636b0bf8bf62d82ae4d980146b5a4df`;
    profile SHA-256
    `f18216e5c08a0ac2eac8986f8338b032c84d2de24a231083ae28083bb061f44f`.
  - Profile-delta proof: removing the three literal package-directory rules
    from v3 reproduces the recorded v2 profile SHA-256
    `7dcef9b4e429752695700817d0748560f6832e2d8b6b1f6f4ec253f2a423a816`.
  - Canary verdicts: `source_drift`, `pickle`, `object_dtype`,
    `oversize_frame`, `extra_frame`, and `malformed_response` pass;
    `file_read`, `file_write`, `environment_secret`, `network`,
    `process_creation`, `fork`, `signal`, `tracing`, `repository_read`,
    `timeout`, `memory_abuse`, `stdout_injection`, `private_fd_injection`, and
    `worker_crash` are `not_verified_in_builder_sandbox` with exact error
    `sandbox-exec: sandbox_apply: Operation not permitted`.
  - `experiments/family_b_target_speed_v1/receipts/builder_runtime_no_learning_smoke.json`:
    canonical SHA-256
    `552e2e39a7438d76af9f7025b6413080d63d791c19be1fa398d8c3a34b257ca0`;
    file SHA-256
    `5cb1d36a64a27a1eb1055af18deb384291de82139048046de61ce1e8fd1b77e5`;
    runtime fingerprint SHA-256
    `d1bf3d270429f1d41d9743f21a6bd090ffadd41e1e500a67b9afeabf5827da50`;
    unchanged eight-step smoke SHA-256
    `048cf64ca8b06f6bd9f00c1b591ba1d0b40fb87620114db6f3a3c367f4218bdc`.
  - `docs/operations/CURRENT_RESEARCH_HANDOFF.md`.
  - Verification: sandbox tests `22 passed, 7 skipped` with exact reason
    `sandbox-exec: sandbox_apply: Operation not permitted`; runtime receipt
    replay `1 passed`; full suite `44 failed, 1143 passed, 9 skipped`, with
    observed failure IDs exactly equal to the saved 44-test baseline; Ruff
    lint and format checks pass. No training or behavioral evaluation ran.
- `TASK-20260904-B0FIX4` repair changed-files subset and receipts:
  - `src/oracle_composition/rewards/_sandbox_worker.py`.
  - `tests/rewards/test_sandbox.py`.
  - `experiments/family_b_target_speed_v1/PROTOCOL.md`.
  - `experiments/family_b_target_speed_v1/receipts/builder_sandbox_canaries.json`:
    regenerated byte-identically; canonical SHA-256
    `3cfc64c0868c606cce9b400cf88493f2b9057d7c7dfe668cfd016b6114bf6238`;
    file SHA-256
    `ba1cb6246298d90f7011ed559c1d1829e636b0bf8bf62d82ae4d980146b5a4df`;
    unchanged profile SHA-256
    `f18216e5c08a0ac2eac8986f8338b032c84d2de24a231083ae28083bb061f44f`.
  - `experiments/family_b_target_speed_v1/receipts/builder_runtime_no_learning_smoke.json`:
    canonical SHA-256
    `69a20c2da8185f6c435be3249f887e557b997255eef5c4a570e345ce0d3d6ce3`;
    file SHA-256
    `2f6276011998c6105ca6714f7b380575f4a1c49e8497aac471c0e35e55629ea6`;
    runtime fingerprint SHA-256
    `26f93064e481c3173315f59401dce0a839681c00da632cd981fb00b7b6a3e9fd`;
    source-tree SHA-256
    `e39d0b08c09813b987c15de0cd8e187a5e2f68c7e9ad964b9c27c00768c39e51`;
    worker-source SHA-256
    `3ab6818d75e5f2d015e9006be9a5a8dc96ff9c43ab823b9f6e0dc496a49ddac9`;
    unchanged eight-step smoke SHA-256
    `048cf64ca8b06f6bd9f00c1b591ba1d0b40fb87620114db6f3a3c367f4218bdc`.
  - `experiments/family_b_target_speed_v1/receipts/host_sandbox_canaries.json`
    remains the pre-B0FIX4 `18/20` profile-v3 diagnostic: canonical SHA-256
    `097006bf8becd4a6b4bed5ffb4dbe56174516e9a878a76a71827ac3d34bf7a97`;
    file SHA-256
    `52bbc11b50513fe3d83bb5c792ebbdf50c35b1069d960a418a8c9b4bcf67780e`.
  - `docs/operations/CURRENT_RESEARCH_HANDOFF.md`.
  - Verification: sandbox tests `24 passed, 9 skipped` with exact reason
    `sandbox-exec: sandbox_apply: Operation not permitted`; runtime receipt
    replay `8 passed`; full suite `44 failed, 1145 passed, 11 skipped`, with
    observed failure IDs exactly equal to the saved 44-test baseline; Ruff
    lint and format checks pass. No training or behavioral evaluation ran.
- Local ignored artifacts created by the slice:
  - `artifacts/bootstrap_tqc_humanoid/external_actor_import_v1.json`: SHA-256
    `ce90c312f7c222847a936edd2d964d386bab01d1acb0fd924de3a8db951430cb`,
    11,308 bytes.
  - `artifacts/bootstrap_tqc_humanoid/farama_minari_humanoid_v5_tqc_actor_v1.npz`:
    SHA-256
    `60987a4e054db2e04f9cb3ab73e13dfe8e2f3ec7dec46346d2b9d0277ad18d9b`,
    618,674 bytes; actor-state fingerprint
    `3fd39cc715a10126fd92b20f6ce213c380eb4d5df843a42315aac50cf116748a`.
  - `artifacts/bootstrap_tqc_humanoid/external_actor_equivalence_v1.json`:
    SHA-256
    `d57eedf35ded4dc5e1f9e6f1cb04c9a67f5db1b491d1825ae6a2196099fc6d32`,
    2,429 bytes.
- Metadata registration receipt: SHA-256
  `5ba0845e8b0cd9b6f39c956ddc46d0f46e8e08690d8bdf832941a1f914a8e0cf`,
  9,395 bytes. LFS siblings are labeled `lfs.sha256`; small files are labeled
  `git-blob-sha1` from `blobId`; each LFS API record retains its `lfs` sub-object.
- Safe-globals receipt: 75 sorted qualified names, SHA-256
  `7b70391289d8e8d285612f5ee4db68739af844eae8d945964d1174c83ce7d4b9`;
  only `builtins`, `traceback`, `collections`, `torch`, and `torch.*` modules;
  unchanged before/after the one weights-only load.
- Test receipts:
  - new focused files: `39 passed in 2.37s`;
  - extended focused set including committed NPZ/equivalence regressions: `81
    passed, 1 skipped in 4.73s`;
  - full suite: `44 failed, 1053 passed, 2 skipped in 112.82s`; its sorted
    failing-node set equals the 44-node baseline at
    `artifacts/bootstrap_tqc_humanoid/sandbox_baseline_failures.txt` (SHA-256
    `c06586e4449fad6b00e6356e6a70e2d75e3476fb8f4fbbf0b2c4e72c547a8345`),
    with no new failures;
  - receipt/NPZ/index revalidation: registration valid, import receipt valid,
    strict NPZ valid, equivalence receipt valid, Git-index policy passed;
  - `.venv/bin/python -m ruff check .`: all checks passed;
    `.venv/bin/python -m ruff format --check .`: 191 files already formatted.
- No training, environment rollout, 20-reset screen, or behavior evaluation ran.

## Current no-go findings

These findings were observed against the uncommitted TQC-v2 working tree. They
remain open on the parked code and no longer sit on the route, but must be
retested if that code ever returns.

| id | blocker |
|---|---|
| TQC-P1-01 | The one-attempt rule is directory-local and can be bypassed with another fresh directory. |
| TQC-P1-02 | A failure after process start can escape complete cleanup and can misreport whether a worker started. |
| TQC-P1-03 | Process-group signal authority can be inferred from the OS before worker self-report is authenticated. |
| TQC-P1-04 | Authenticated worker failures before manifest acknowledgement can be downgraded to generic integrity failures. |
| TQC-P1-05 | Signal and terminal-receipt ownership is not safe across every public launch path. |
| TQC-P1-06 | A transient macOS `EPERM` group probe can false-reject cleanup instead of remaining indeterminate within the deadline. |
| TQC-P1-07 | The training-to-persistence semantic bridge for policy, entropy, and optimizer state is unresolved. |
| TQC-P1-08 | Success does not yet revalidate every retained preflight, manifest, reservation, and worker receipt. |

Additional constraint:

- A production-ID canary would consume the sole attempt. Any canary must use a
  separate, explicitly non-authorizing identity. This constraint lapses with
  the one-attempt rule if ADR 0005 survives review.

## Orchestration state

| component | observed state |
|---|---|
| Claude Code | `2.1.260`; session `807bcdb2…` recorded `claude-fable-5-1` at startup; hooks accepted every tool call |
| Fable policy | Exact model requested at launch; `switchModelsOnFlag=false`; Samuel's `/status` pending |
| session guard | Project hooks deny requested non-Fable switches and all Claude tool use when the recorded model is not Fable or active effort is not `max` |
| Codex | Official standalone CLI at `~/.local/bin/codex`; `.codex/config.toml` requests `gpt-5.6-sol` at `max` with `multi_agent=false` |
| worker path | `scripts/start-sol-worker` made its first two real launches at `16:21Z`, both read-only, both in detached screen sessions with compact status |
| write control | Fable held the strategy lease for the audit and released it at the end of the startup turn; the recorded scope is an audit boundary |
| continuity | The 30-minute heartbeat is coordination-only; no takeover record exists |

## Resume order

1. Verify the 03BFIX lease and inspect the complete builder diff plus
   `corpus_run_v2_validation.json`.
2. Recheck the v2 run-level hashes and the immutable v1 hashes before any Git
   write.
3. Commit the builder slice and documentation/integration changes separately.
4. Record the failed expert screen in strategy and ADR 0005 without changing
   the frozen gate or claiming a tracker admission.
5. Fold any P1 findings from the two read-only 03B reviews into a bounded
   follow-up slice.
6. Launch the already-planned importer closure `03A2FIX2`, then consume the
   packet-B design survey.
7. Do not train, retry the screen, or widen the v2 claim ceiling.
8. Update this file at every control transfer and at least every 30 minutes
   during active work.

## Handoff update contract

- Keep the three-row table first.
- Replace, do not append to, its current statements.
- Separate requests, implementation, and measurements.
- Name changed files, test receipts, remaining blockers, active worker/thread
  IDs, and the next bounded action.
- Never convert a green test, UI launch, or zero-action smoke test into a
  behavioral claim.

## Git-state rule observed 2026-09-04

The Codex `workspace-write` sandbox denies writes under `.git`. Sol workers
cannot branch, stage, commit, stash, or create worktrees. Fable performs every
git state change under its own lease after review. Builder 03A stopped at that
boundary at `16:52Z` with all preconditions verified; Fable completed the WIP
preservation at `2026-09-04T16:58Z` with `artifacts/bootstrap_tqc_humanoid/wip_preservation_fable_verification.txt`
as the byte-level receipt.

## Reward-track design adopted `2026-09-04T17:10Z`

Survey `TASK-20260904-04` (thread `01a06d48-1482-73f2-800a-a3bb69d9de5e`,
run `.orchestration/sol-runs/20260904T163715Z-ddddde81-0085-49fd-8714-0d512c9d39a0`)
delivered the `family-b-target-speed-v1` design. ADR 0006 records it. Builder
packet `TASK-20260904-B0` implements the no-training first slice. Review
packets `06` and `07` are written for the 03A2 diff.

## Baseline suite receipt `2026-09-04T17:13Z`

`.venv/bin/python -m pytest -q -p no:cacheprovider` on clean `main` at
`48955df`, outside the Sol sandbox: `1060 passed, 2 warnings in 122.17s`
(`17:08:13Z` to `17:10:16Z`). Software behavior only. Inside the Sol sandbox
the same tree shows `44 failed` for environment reasons (denied socket binds,
CPU fingerprint `arm`); builders record their own in-sandbox baseline and must
not add failures.

## Tracker-track design adopted `2026-09-04T17:13Z`

Survey `TASK-20260904-05` (thread `01a06d4c-b071-7493-863b-fbcdaafa6c5d`,
run `.orchestration/sol-runs/20260904T164217Z-62e441c4-8938-4986-b267-cb0f8d21ce80`)
delivered the same-runtime reference, Tier-D, E3, residual-PPO E4, and E5
designs. ADR 0005 records the chain. Packets `03A3` and `03B` implement the
first two slices; packets for E4 and E5 follow their reviews.

## Builder 03A2 diagnostic `2026-09-04T17:45Z`

Fable ran the worker's stages in a separate process on the pinned expert
`policy.pth`: resource limits applied (macOS reports no finite `RLIMIT_AS`),
then category `process safe globals refused`. A fresh interpreter on Torch
`2.14.0` reports `75` pre-registered safe globals (builtin exception classes,
`traceback.FrameSummary`, Torch internals). The corrected invariant is
"no additions during the load and no entry outside `builtins`, `traceback`,
`collections`, or `torch`", recorded by count and SHA-256 in the receipt.
Packet `TASK-20260904-03A2R` carries the fix and the remaining steps. The seven
uncommitted code files from launch 3 stay in the working tree for it.

## External actor import receipts `2026-09-04T18:13Z`

Builder `03A2R` completed the slice. Local ignored artifacts under
`artifacts/bootstrap_tqc_humanoid/`: registration receipt SHA-256
`5ba0845e8b0cd9b6f39c956ddc46d0f46e8e08690d8bdf832941a1f914a8e0cf`; import
receipt `ce90c312f7c222847a936edd2d964d386bab01d1acb0fd924de3a8db951430cb`;
strict actor NPZ `60987a4e054db2e04f9cb3ab73e13dfe8e2f3ec7dec46346d2b9d0277ad18d9b`
(`618,674` bytes); equivalence receipt
`d57eedf35ded4dc5e1f9e6f1cb04c9a67f5db1b491d1825ae6a2196099fc6d32`;
actor-state fingerprint
`3fd39cc715a10126fd92b20f6ce213c380eb4d5df843a42315aac50cf116748a`; safe
globals `75` entries, hash `7b70391289d8e8d285612f5ee4db68739af844eae8d945964d1174c83ce7d4b9`.
In-sandbox suite: `44 failed, 1053 passed, 2 skipped`, failures identical to
the frozen baseline; new focused tests `39 passed`; ruff lint and format
passed. Evidence class `external_base_import`, an `interface_check`. Nothing
behavioral ran.

## Review round on `1d6b461` and B0 host checks `2026-09-04T20:21Z`

- Scientific review `06` (thread `01a06da2-b496-7cb2-b89a-c87725723b53`): `ACCEPT-WITH-REPAIRS`, `0` P0, `3` P1 (`SCI-03A2-01` nominal payload fingerprint; `-02` sorted safe-globals comparison; `-03` vacuous negatives). Repair packet `TASK-20260904-03A2FIX` also carries the storage-interval overlap check found by the adversarial probe.
- Adversarial review `07` (thread `01a06da2-b54c-70a1-8ffb-c2a2f033d1e6`): `turn.failed`, flagged by the Codex cyber-safety filter after constructing one overlap probe. Future adversarial packets are phrased as defensive robustness reviews without payload construction (`TASK-REVIEW-ROBUSTNESS-GENERIC`), driven by `.orchestration/review-target-adv.txt`. A matching `TASK-REVIEW-SCI-GENERIC` reads `.orchestration/review-target-sci.txt`.
- B0 host checks: Seatbelt bootstrap passes on the host; `RLIMIT_AS 1 GiB` raises `ValueError` on macOS; `14` OS canaries fail in pre-exec, `6` protocol canaries pass; host receipt `experiments/family_b_target_speed_v1/receipts/host_sandbox_canaries.json` (canonical SHA-256 `62c2529fd0430b0d9bd9fe7c0920015535961aa9771874aa65f0cecfad161190`); outside-sandbox suite `7 failed, 1179 passed`, all seven in `tests/rewards/test_sandbox.py`; ruff clean. Repair packet `TASK-20260904-B0FIX`.

## B0 host diagnosis `2026-09-04T21:08Z`

After B0FIX, the host run of the OS canaries still failed with
`worker pipe closed before a complete response`. Direct execution of one
canary showed (1) `ModuleNotFoundError: oracle_composition` under a permissive
profile because the runner resolves the venv symlink to the base uv
interpreter, which has no venv context under `-I`; and (2) `SIGABRT` at
startup under the v1 profile, fixed only by `(allow file-read-metadata)` plus
`(allow file-read-data (literal "/"))` with data reads under `/Users`. Leave-one-out
over every other top-level area left the worker running. Packet
`TASK-20260904-B0FIX2` carries both repairs. The reward-track worker's exit
code `64` was observed on the host and must be documented by the builder.

## B0 host diagnosis, second round `2026-09-04T21:42Z`

Under profile v2 with the venv interpreter, the worker starts and reports
`ModuleNotFoundError: No module named 'oracle_composition.rewards'`. Inside the
sandbox `oracle_composition` resolves as a namespace package from
`.venv/lib/python3.13/site-packages/oracle_composition/` (the force-included
`_runtime/uv.lock` resource) because listing `src` is a data read the profile
does not grant. Three literal directory grants (`src`,
`src/oracle_composition`, `src/oracle_composition/rewards`) let the regular
package win; the worker then runs to its documented exit paths. Packet
`TASK-20260904-B0FIX3` applies them as profile v3.

## B0 host verification under profile v3 `2026-09-04T21:58Z`

Host canaries: `18` passed, `2` failed (`network`, `tracing`, exit `4`);
`memory_abuse` passed with mechanism `timeout_kill`. Host receipt canonical
SHA-256 `097006bf8becd4a6b4bed5ffb4dbe56174516e9a878a76a71827ac3d34bf7a97`.
The two failures are probe defects: socket creation and `getpriority` are not
operations Seatbelt governs. Packet `TASK-20260904-B0FIX4` replaces them with
`connect`/`bind` and `libproc` probes. No profile change is needed.

## Two-lane orchestration `2026-09-04T22:15Z`

Samuel authorized concurrent Fable and Astra orchestration. ADR 0007 records
the lanes. Mailbox: `/Users/samueldoane/Documents/ChatGPT/humanoid-harness/.git/harness-coordination`
through the Astra checkout's `scripts/research-mailbox` (`inbox fable`,
`ack fable ID`, `send --from fable --to astra ...`, `status fable`, `board`);
Fable writes to it under its lease with that directory in the audit scope.
Acceptance `20260904T221414.221774Z-a293af4af0eb4362a66d1a71554d3c15` replied
to proposal `20260904T201259.759134Z-ee5e9a7d67154a118dac6ffffd73e4c1`. Poll at
turn start, before dispatch or integration, after each worker, and about every
30 minutes. B0 ownership transfers only by a `handoff` message after its two
reviews. Heavy jobs over ten minutes need a mailbox proposal and a peer reply.

Current B0 host record (supersedes every earlier canary statement in this
file): profile v3, `20/20` canaries passed, `memory_abuse` mechanism
`timeout_kill`, host receipt canonical SHA-256
`5465426251311c147843488f8b92eb7b0fca0428fee4091267f41dde53adfdd9`, sandbox
tests `33 passed`, full suite `1200 passed`. Software-boundary evidence only:
B0 ran no training, learned-policy rollout, protected endpoint measurement,
or humanoid behavioral evaluation. Earlier sections describing `14 failed`,
`18/20`, or receipt `097006bf…` are superseded history.

## B0 reviews and handoff `2026-09-05T01:03Z`

Scientific review `09` (run `20260904T221620Z-cd03d6f4-3479-4c26-9410-ecd4c2d57dcd`) and robustness review `10`
(run `20260904T221621Z-d07a5b95-5955-462f-834e-61f108bda886`) of `89d4c34`: both `ACCEPT-WITH-REPAIRS`, no P0,
`6` and `10` P1. Shared findings: candidate code executes in the parent during
validation and calibration; admission and result lineage unbound; canary
ledger contradiction (the protocol still cites the pre-repair `18/20` receipt);
canaries that count any exception as containment. Scope note: the B0 commit
also carried ADR 0007 and the decisions index; recorded as an explicit
exception. A `handoff` message to Astra transferred the reward paths with all
findings and the condition that nothing executes or trains on B0 until the P1
repairs land and are re-reviewed. Fable remains promotion owner for `main`.

Importer repair `03A2FIX` (run `20260904T221621Z-0d2563da-2794-4f84-a220-bb7fd4525ab5`, thread
`01a06e7e-89f4-7fd2-b4cd-e9f93bde1e8d`): all ten findings implemented,
`173 passed, 1 skipped` focused; import receipt schema v2
`c2379a21079c40928548b593cb86a8fd8eea468787cf31c28481792d3121e80b`,
equivalence receipt v2 `0d5ec467051e2eebc16a0a186afbe85cc1364afbeab8fcbefa41f088cb656f31`;
NPZ and actor fingerprint unchanged. One full-suite failure outside the
baseline: the reward runtime receipt's whole-tree `source_tree_sha256`
(recorded `e39d0b08…`, current `7496476d…`), regenerated by Fable once and
recorded as a reward-lane P1.

Samuel's alignment question (via Astra, `20260904T221845…`) answered with an
independent interpretation of reference composition; recorded in the strategy
as the frame for Experiment 003: a three-gait composition task with
state-triggered transitions and recovery, LLM-designed oracle program,
feedback-driven revision, elapsed-time and hand-written baselines.

## Provenance correction for `425d3bc` and integration rule `2026-09-05T03:13Z`

Commit `425d3bc` bundled two integration-owned changes with the builder's
packet slice: the strategy addition (Experiment 003 composition frame) and the
regenerated reward runtime receipt
`experiments/family_b_target_speed_v1/receipts/builder_runtime_no_learning_smoke.json`.
Both were authored by Fable under its lease, outside packet `03A2FIX`'s
allowlist; the packet's changed-files ledger and the earlier "scope-only"
wording were therefore inaccurate. Reviews `11` and `12` flagged this
(`SCI-03A2FIX-01`, `R-07`). Rule from here on: builder slices and Fable's
integration edits are committed separately, and every integration edit is
listed in the handoff with its authority. Host verification results are now
recorded durably in `experiments/bootstrap_tqc_humanoid/reviews/fable_host_verification.json`.

## Importer hardening synthesis `2026-09-05T03:13Z`

Robustness review `12` of `425d3bc` (run
`20260905T010724Z-6ed23ba1-3467-46d6-8d8b-ce4cf60933be`): `R-01` rights set from
every registered payload; `R-02` release-bundle scanning; `R-03` bounds before
allocation; `R-04` parser exception normalization; `R-05` installed-file
identity; `R-06` atomic publication; `R-07` provenance (fixed above); `R-08`
boundary-level negative tests; P2 `R-09`, `R-10`. Fable accepts `R-01`,
`R-03`, `R-04`, `R-06`, `R-08` in full, narrows `R-02` to a git-index
guarantee until a release pipeline exists, binds locked artifact hashes and the
Python ABI for `R-05`, and closes the loop with one scoped closure review.
Packet `TASK-20260905-03A2FIX2` carries this; it runs after `03B`.

## Why the open importer P1s cannot alter the corpus's NPZ chain `2026-09-05T05:18Z`

Astra asked for this before treating the corpus as admitted evidence. The
corpus consumes only the three strict NPZ files and their documented hashes
(`PUBLIC_EXPERT_IMPORT.md`); it never calls the importer. Each NPZ was produced
from pinned `policy.pth` bytes (SHA-256 verified against the Hugging Face LFS
record), through a single weights-only load, by copying eight float32 tensors
whose shapes, contiguity, and finiteness were checked, and its actor outputs
match a locally reconstructed actor bitwise on the fixed batch. The open
findings concern the importer's failure paths and evidence packaging, not the
success path that copied those bytes: `R-01`, `R-02` (rights coverage and
release scanning), `R-03`, `R-04` (bounds and exception normalization),
`R-06` (atomic publication), `R-08`, `T-1` (test coverage), `V-1`
(validation manifests), `E1-B`, `E1-C` (receipt revalidation and all-files
verification). `R-05` narrows the reproducibility claim of the equivalence
receipt to same-host, same-lock; it does not change the copied tensors. The
zero-residual E1 gap (`E1-A`) concerns the tracker contender, which the
corpus does not use. None of them can change an NPZ byte or its hash; the
corpus certificates bind those hashes, and a later importer change that
altered the bytes would break that binding.

Corpus compute bounds, for the protocol: about `128` clips of `1,000` control
steps for frozen-actor rollouts plus separate-process replay certification and
E3 forks; minutes of CPU on this host; recorded wall time in the validation
manifest. The 03A3 review state is `ACCEPT-WITH-REPAIRS` and E1 remains an
interface check.
