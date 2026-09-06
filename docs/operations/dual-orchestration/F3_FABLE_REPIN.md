# F3 Fable-lane re-pin

| status | current truth |
|---|---|
| progress | The accepted F3 initial-only protocol is re-pinned from Astra to Fable's `main` checkout; 117 focused tests and three-file Ruff checks pass. |
| bottleneck | No candidate call, candidate admission, experimental simulator run, training, protected evaluation, measured feedback, or reward result exists. |
| next step | Fable reviews and commits this static re-pin, then freezes the separate matched T2 reward-study protocol. Do not dispatch the F3 candidate call from this slice. |

## Builder boundary

| item | exact value |
|---|---|
| checkout / branch | `/Users/samueldoane/Documents/ChatGPT/humanoid-harness` / `main` |
| base commit | `8d617e30dd239529a42e3a0211314d6173ffeafd` (clean at launch) |
| accepted Astra F3 source | `dc402e5d7a42af3bdd1053b349815ed30f34a859`, merged before this slice |
| import origin | `/Users/samueldoane/Documents/ChatGPT/humanoid-harness/src/oracle_composition/__init__.py` |
| writer lease | `CLAIMED` by `sol-builder-20260906-f3pin`; model `gpt-5.6-sol`; role `builder`; exact launch-note scope |
| Git writes | none; sandbox denies `.git` writes |
| execution boundary | no candidate dispatch, standalone simulator/experiment command, training, or network call; validation commands only |

The lease, branch, clean tree, base commit, and import origin were checked before
editing. The launcher owns lease renewal and release. Historical Astra result,
repair, and acceptance records remain unchanged.

## Re-pinned expected configuration

| expectation | accepted Astra binding | Fable binding |
|---|---|---|
| checkout | `humanoid-harness-astra` | `/Users/samueldoane/Documents/ChatGPT/humanoid-harness` |
| branch | `astra/reward-loop` | `main` |
| package import | Astra checkout `src/oracle_composition/__init__.py` | exact Fable checkout import origin above |
| preparation / ingestion | Astra parent lease | owner `fable-f3-prepare`, role `builder`, scope `t2-initial-packet-preparation-and-ingestion-only` |
| candidate call | owner `astra-f3-initial` | owner `fable-f3-initial` |
| candidate role / scope | `candidate` / `t2-initial-parameter-hypothesis-only` | unchanged |
| candidate runner request | `gpt-5.6-sol`, `max`, `screen`, read-only | unchanged |

Packet records, dispatch intents, and ingestion receipts carry these values only
as `expected_*` configuration with semantics
`expected_configuration_not_served_model_attestation`. Raw retained launcher
bytes remain the evidence for what a future request says; neither the expected
configuration nor a locally consistent envelope authenticates the served model.

## Baseline and F2 binding

| artifact | exact identity |
|---|---|
| accepted baseline reference | commit `87d2e39c47b6747b505bc2657d505aeda2265b5e` and `tracking_only_v1.json` |
| re-pinned baseline reference | commit `8d617e30dd239529a42e3a0211314d6173ffeafd` and the same path |
| current Git blob object | `8faef6740606769ed5e3fa11014b583ad29f0365` |
| baseline bytes | 1,773; final byte `}`; no newline |
| baseline SHA-256 | `eea2b6a9893e6e4ca5aea5d6787580f12062084e758cc9c27db2f1376bcb1c5f` |
| F2 evaluator source | `064393887bb4a7157d614981cc2000940252e12c31ad0aabc13109737fd94301` |
| F2 recipe/parser source | `c60dea03e6f0e71b81875fea59c84bd8fe00ce39f94ac7c54ca6e2a57360dccb` |
| T2 input source | `9607f2d56a54922eac06ec7fcc740b79e68d4ed9e88b192602aae2900fb3f0d2` |
| canonical F2 bounds | 137 bytes / `2c6030264231a52320a44fe0f4d4ed519bc2e635b892f1b9c0d8929166106933` |

FT2R3 did not change the baseline bytes. The accepted and re-pinned commit
references resolve to the same baseline content. Its admission fields remain a
declared configuration, not evidence of successful runtime loading.

## Deterministic initial dossier

| item | exact identity |
|---|---|
| canonical packet record | 25,231 bytes / `464860f284213c5aac193a7914f7a90b6ec160c6ea0a93f7b83d209ddbdcb3ff` |
| rendered prompt | 6,011 bytes / `4d1a29765d929472e2f2a346294f98ac08c0e35d3e18c147e761e46678fe2bbc` |
| semantic request | `40e3306985e28296fdc24264e857008cda4367f940e9bf71649f42b5fb4e1274` |
| T2 contract bundle | `b5f9cd440b269090a1573fc3a540ce4c86e7ddcd0b70510402210fa55c2bde60` |
| missing-evidence dossier | `32cdcc548cefedcabee304d75d5d84c8dcbe9e2c2129cee541692380c90f890b` |

The rendered bytes state the T2 task—hold `3.0 m/s` COM forward speed from the
expert start—the exact baseline identity, F2 family and four source/bounds
identities, and that only bounded `alpha` and `beta` are authorable. They state
that no measured feedback exists. Regression checks exclude retained payload
bytes, simulator-state fields, corpus worker inputs, and the phase-B execution
manifest identity.

## Changed F3 artifacts

| path | accepted SHA-256 | re-pinned SHA-256 |
|---|---|---|
| `src/oracle_composition/reward_search/t2_model_contracts.py` | `4f1e7d36797f2d3d2e25050bbd980fa4c0c1f65949cacc931a73fc1dac208904` | `3de98c68fda1714e22e165f84f7b57abcf89963bfb60a8c391cd74973c629482` |
| `src/oracle_composition/reward_search/t2_model_protocol.py` | `e7390f16891f240435a1fe15d247bc58483a414e325da6db4fe2d268c74d1a94` | `6d192c804fe6f59233e609926a5054ca7b07e511d0ae8b51f3724f4b433f558a` |
| `tests/reward_search/test_t2_model_protocol.py` | `e59172944aa1e7500126236c357bbdb46f624555c165df707e59791216a40618` | `e514bdeffafc133d355d6dd40d21a89e717595e7c4e3a815b87dca53cf29137a` |
| `docs/operations/dual-orchestration/F3_ONE_CALL_PROTOCOL.md` | `193ddea8f3dbc363305157f8097daebdc5a186adcc605d50592bd853ffad89d8` | `1c31701ae184a02d15ef3d47668613861ec128ac5137534934b4eb44472f08bd` |

The new dispatch-intent contract binds one initial call, zero remaining
attempts, zero revisions, zero retries, and a 1,200-second elapsed-duration
deadline measured from immutable `request.json.created_at_utc`. Its fixed
`dispatch-intent.json` publication is no-overwrite. Request, task packet,
terminal result, and final response retention; bounded reads; exact-type public
ingress; all-or-none identities; and hypothesis-only acceptance are unchanged.

## Verification

| check | result |
|---|---|
| Focused F3 plus F1 predecessor tests | `117 passed in 0.63s` (the accepted 112 plus five new checks) |
| New negatives | Astra owner rejected; Astra checkout receipt rejected; mismatched baseline receipt rejected; second dispatch intent refused unchanged |
| Ruff lint, three changed Python files | passed |
| Ruff format check, three changed Python files | passed |
| Full suite excluding the named reward-lane receipt test | `1757 passed, 18 skipped, 1 deselected, 44 failed in 220.39s`; all 44 failures exactly match the sandbox-baseline node-ID ledger |
| `git diff --check` | passed |

## Claim ceiling

This slice establishes Fable-local expected configuration, deterministic packet
content, and duplicate-intent refusal. It is not a reward result. It establishes
no candidate service call, model authentication, candidate admission, simulator
behavior, policy training, protected evaluation, measured feedback, reward
improvement, or humanoid competence.
