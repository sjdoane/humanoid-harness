# F3 Fable-lane re-pin

| status | current truth |
|---|---|
| progress | T2PAIRR1 binds repaired exact-byte validation and production routing into receipt `1a2b7ece…2348`, pending study `a839362a…eecb`, and F3 seal `6ce006bb…5a9d`. |
| bottleneck | Dispatch remains `withheld_pending_dispatch_verdict`; the canonical candidate is TBD and T2PAIRR1 needs narrow independent review and commit. No candidate call or behavioral result exists. |
| next step | Astra reviews the T2PAIRR1 delta. Only after acceptance, commit, and a separate dispatch verdict may Fable regenerate a call packet; candidate admission and its clean-HEAD re-seal remain later gates. |

## Original F3PIN builder boundary

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

## Commit-bound F3PIN checkpoint and lockfile

| field | exact value |
|---|---|
| `builder_launch_base` | `8d617e30dd239529a42e3a0211314d6173ffeafd` |
| `target_parent` | `c0332b1148a86a715d35aa62c73f3affc6178715` |
| `target` | `b071fc553d378d0fe6a1c4304fc4055cdff7b4ea` |
| complete target path list | `README.md`; `docs/operations/dual-orchestration/F3_FABLE_REPIN.md`; `docs/operations/dual-orchestration/F3_ONE_CALL_PROTOCOL.md`; `src/oracle_composition/reward_search/t2_model_contracts.py`; `src/oracle_composition/reward_search/t2_model_protocol.py`; `tests/reward_search/test_t2_model_protocol.py` |
| `uv.lock` | SHA-256 `81b92d15dd2da62f27cd770322db78008d5387b530dc71e053f0d56b327f0b40`; unchanged at `8d617e3`, `c0332b1`, `b071fc5`, and F3PINR1 launch base `c969009` |

The launch base and target parent are different provenance facts: T2 protocol
commit `c0332b1` landed between the F3PIN builder launch and Fable's integration.
Target `b071fc5` omitted the packet-required current-handoff update. This record
repairs that audit gap without rewriting the historical target. Going forward,
Fable integrates the handoff update in the same slice commit as the code and
protocol it describes.

## F3PINR1 builder boundary

| item | exact value |
|---|---|
| checkout / branch | `/Users/samueldoane/Documents/ChatGPT/humanoid-harness` / `main` |
| launch base | `c969009` (clean at launch; T2A committed as `3b14d54`) |
| writer lease | `CLAIMED` by `sol-builder-20260906-f3pinr1`; model `gpt-5.6-sol`; role `builder`; exact launch-note scope |
| pairing status at launch | Astra proposal `20260906T055257` pending; no acceptance receipt |
| Git writes | none; sandbox denies `.git` writes |
| execution boundary | no candidate dispatch, training, standalone simulator command, or network call; validation commands only |

## Re-pinned expected configuration

| expectation | accepted Astra binding | Fable binding |
|---|---|---|
| checkout | `humanoid-harness-astra` | `/Users/samueldoane/Documents/ChatGPT/humanoid-harness` |
| branch | `astra/reward-loop` | `main` |
| package import | Astra checkout `src/oracle_composition/__init__.py` | exact Fable checkout import origin above |
| preparation / ingestion | Astra parent lease | owner `fable-f3-prepare`, role `builder`, scope `t2-initial-packet-preparation-and-ingestion-only` |
| candidate call | owner `astra-f3-initial` | owner `fable-f3-initial` |
| candidate role / scope | `candidate` / `t2-initial-parameter-hypothesis-only` | `candidate` / same prefix plus `;call_identity_sha256=<canonical digest>` |
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

## Original deterministic initial dossier (`b071fc5`)

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

## Original F3PIN changed artifacts (`b071fc5`)

| path | accepted SHA-256 | re-pinned SHA-256 |
|---|---|---|
| `src/oracle_composition/reward_search/t2_model_contracts.py` | `4f1e7d36797f2d3d2e25050bbd980fa4c0c1f65949cacc931a73fc1dac208904` | `3de98c68fda1714e22e165f84f7b57abcf89963bfb60a8c391cd74973c629482` |
| `src/oracle_composition/reward_search/t2_model_protocol.py` | `e7390f16891f240435a1fe15d247bc58483a414e325da6db4fe2d268c74d1a94` | `6d192c804fe6f59233e609926a5054ca7b07e511d0ae8b51f3724f4b433f558a` |
| `tests/reward_search/test_t2_model_protocol.py` | `e59172944aa1e7500126236c357bbdb46f624555c165df707e59791216a40618` | `e514bdeffafc133d355d6dd40d21a89e717595e7c4e3a815b87dca53cf29137a` |
| `docs/operations/dual-orchestration/F3_ONE_CALL_PROTOCOL.md` | `193ddea8f3dbc363305157f8097daebdc5a186adcc605d50592bd853ffad89d8` | `1c31701ae184a02d15ef3d47668613861ec128ac5137534934b4eb44472f08bd` |

Those hashes describe the reviewed F3PIN target, not the later repair below.

## F3PINR1 repairs

| finding | repair |
|---|---|
| F3PIN-SCI-01 | Schema-v4 packet, intent, and call receipts bind the digest of one `call-identity.json`. Its `call_id` derives from the fixed protocol ID, study-manifest SHA-256, and prompt SHA-256. Only the canonical intent path is accepted. The exact launcher `scope` carries the identity digest into retained request/result bytes. Ingestion requires the canonical identity/intent and exclusively creates `ingestion-claim.json`; second runs and same-run provider replays retain rejection receipts. |
| F3PIN-SCI-02 | New `t2_seal_v1.json` binds and re-verifies the expert-hold oracle, T2 training design, evaluator design, study manifest, recomputed pairing key, and tracking-only baseline before prompt rendering. The seal is non-model-facing; the prompt remains exactly 6,011 bytes / `4d1a2976…e2bbc`. It honestly records `pairing_receipt: pending`, so dispatch is withheld. |
| F3PIN-SCI-03 | The commit-bound table above distinguishes launch base, target parent, target, and all six target paths; this handoff is in the repair slice. |
| F3PIN-SCI-04 | The unchanged `uv.lock` SHA-256 is recorded above. |

### Re-seal after Astra scoring integration

Main merged Astra's accepted settling-band scoring change (`3406bb5`) as
`ed9f1d3`. Because the F3 seal transitively binds `phase_b/report_v2.py`, the
pre-merge identities were intentionally invalidated and re-sealed before the
T2AR1 repairs:

| identity | superseded | re-sealed at `ed9f1d3` |
|---|---|---|
| evaluator design | `d8f54f80051e4e3e58a540fc1414e9a971c9e299d903c2a00162bfebaa64a04b` | `7339b06f271fdb053ae0c53d287b92c8b6cbd05fb67f18a33a8ab01cdfb4dbd7` |
| study manifest | `7aefaafcef6d319dfd991b6a16eccc5f5578651d6ac8ebb3573358cc307f0738` | `55b0b6e60216806ff873b21010a8a8eacd1c27570da24dace611fa1c389e1cd4` |
| study pairing key | `4af9c9539bb2c7f9720f3e98a3f195c2b297d6bfe6adc51a1d63e7a8e4ad2464` | `fb2fff65ef5cdbc95a52d88c90a4ec9f22d62d1ba7092abd37b0bff956adb053` |
| adapter receipt | `01c591c566554c386fbfa38cdfc1019579e20af4a3c9a3e72dca958b09fed1f1` | `a3b5baa94fbdb818e02d77f84c775a5a2c647754d65b9570d166401ab377d27d` |
| F3 seal | `8c04f0c2bc9222f9918829f1e5d922c948583ea03da0eb0174a03e2dfcb314eb` | `38dcab8d4f6b52c47c1981a4b2347f1c15d66a1f2ff17729832a25678bbda448` |

No candidate call, smoke, training, or evaluation run used the superseded
seal chain.

### T2AR1 re-seal

T2AR1 adds `t2_execution_manifest_v1`, verified-chain trace lineage,
reward-telemetry separation, reachable invalid-action scoring, and raw-trace
report replay. Those source-bound changes intentionally supersede the
post-Astra seal chain:

| identity | post-Astra seal | T2AR1 seal |
|---|---|---|
| execution manifest | absent | `d675b1ac02ad8713d995bd795ce2130acf41bc7a6fcc0a164167b24e3684ab3e` |
| evaluator design | `7339b06f271fdb053ae0c53d287b92c8b6cbd05fb67f18a33a8ab01cdfb4dbd7` | `7ae812d43c524285941cd567ce1663ff023cb6307229d9472a6dfe6577ebf5b3` |
| study manifest | `55b0b6e60216806ff873b21010a8a8eacd1c27570da24dace611fa1c389e1cd4` | `4eb3440b943355b8eee96e7663a4d542833464a8720d4b8ac102c050d9623627` |
| study pairing key | `fb2fff65ef5cdbc95a52d88c90a4ec9f22d62d1ba7092abd37b0bff956adb053` | `fd91156a949a4484b497a112327db864c2a4cbcbaf0cf1cf5db75064f6a5b3e0` |
| adapter receipt | `a3b5baa94fbdb818e02d77f84c775a5a2c647754d65b9570d166401ab377d27d` | `6bd6f5ab3eb33ea563fff06c01828781d4c01864b7f0384108bfa5161c02540a` |
| F3 seal | `38dcab8d4f6b52c47c1981a4b2347f1c15d66a1f2ff17729832a25678bbda448` | `c0edc94a71d7e7cd23723e0e58b352568c2610d064db170a47a83869f02394ae` |

No run used either superseded chain. The T2AR1 seal is 1,613 bytes and binds
the execution manifest as a sixth immutable input. It still records
`pairing_receipt: pending`; this is a preparation repair, not dispatch
authority or execution evidence.

### T2AR2 re-seal after T2PAIR

The accepted 9,770-byte pairing receipt at
`e5351c3b49ba97cc362ccd68a0bcf797c5077fb0c2ace78535b24e5567e0a259`
is now bound by the pending study and F3 seal. It was not regenerated: its
`4eb3440b…3627` source-study hash and `fd91156a…b3e0` pairing key remain the
explicit predecessor lineage. The post-pairing common-field re-seal is:

| identity | T2AR1 | T2AR2 |
|---|---|---|
| execution manifest | `d675b1ac…b3e` | `9492cc63…e368` |
| evaluator design | `7ae812d4…f5b3` | `634ea93e…2708` |
| study manifest | `4eb3440b…3627` | `8e81792a…2d10` |
| study pairing key | `fd91156a…b3e0` | `64529d78…c1e7` |
| F3 seal | `c0edc94a…94ae` | `924769fe…059e` |

The T2AR2 seal is 1,671 bytes. It records the receipt digest and
`withheld_pending_dispatch_verdict`; it is preparation evidence, not call
authority. The updated deterministic packet is 28,461 bytes at
`6f779d37…d3cc`, with call ID `eeb24a67…4aef` and call-identity digest
`1ae0e967…892d`; the model-facing prompt remains exactly 6,011 bytes at
`4d1a2976…2bbc`.

Report hashes named files and replays traces; cross-manifest reconciliation is
enforced by the final-ready gate. Baseline is registry-resolved; the candidate
resolver is unit-tested and the canonical candidate is TBD. The execution
manifest records launch-base provenance; the execution commit is verified at
admission. Action-bound failure is interface-reachable; no production T2 trace
producer has run.

A second claim of the canonical call identity is refused; repository state
cannot prove the absence of an unrecorded external call.

### T2PAIRR1 re-seal after pairing review

The repaired 9,770-byte pairing receipt is
`1a2b7ece139974117fd5c75e040d9cc52cd51a4e42c9b5afd794a60b02232348`.
Its bytes changed and retain the T2AR2 study and pairing identities as explicit
predecessor lineage. The pending identities changed from T2AR2 as follows:

| identity | T2AR2 | T2PAIRR1 |
|---|---|---|
| execution manifest | `9492cc63…e368` | `7e303034…8d1c` |
| evaluator design | `634ea93e…2708` | `634ea93e…2708` |
| study manifest | `8e81792a…2d10` | `a839362a…eecb` |
| study pairing key | `64529d78…c1e7` | `affe8347…ec19` |
| F3 seal | `924769fe…059e` | `6ce006bb…5a9d` |

The prior T2AR2 packet identity is superseded by the changed seal. This repair
did not render or claim a replacement packet and did not make a candidate call.
Any later packet must be prepared from the accepted, committed T2PAIRR1 bytes
under the one-call preconditions.

### F3PINR1 immutable identities

| item | bytes / SHA-256 |
|---|---|
| T2 pre-dispatch seal | 1,400 / `8c04f0c2bc9222f9918829f1e5d922c948583ea03da0eb0174a03e2dfcb314eb` |
| schema-v4 packet record | 28,101 / `ce166b7a655665990574c7452384d5245490f97f83f2c150ea69157117b5e199` |
| unchanged model prompt | 6,011 / `4d1a29765d929472e2f2a346294f98ac08c0e35d3e18c147e761e46678fe2bbc` |
| study manifest | 5,474 / `7aefaafcef6d319dfd991b6a16eccc5f5578651d6ac8ebb3573358cc307f0738` |
| derived `call_id` | `570ebdc2f117e80b02b269b480045a34df57ed63fb9becb4620a5c6be5b54254` |
| canonical call-identity digest | `23b45b7b0d5e5abc01a21a48d89828e1b85dd83cb41c473ab495b44683f17cc6` |
| launcher scope | `t2-initial-parameter-hypothesis-only;call_identity_sha256=23b45b7b0d5e5abc01a21a48d89828e1b85dd83cb41c473ab495b44683f17cc6` |

| changed artifact | F3PINR1 SHA-256 |
|---|---|
| `src/oracle_composition/reward_search/t2_model_contracts.py` | `07ca186b63b827764283640bf435135c07cc83646f62f45583d8459e7f070b68` |
| `src/oracle_composition/reward_search/t2_model_protocol.py` | `dea23b787110afc97e22c09ee7ae781fd0f03ebe123e616c474551efcae8c429` |
| `tests/reward_search/test_t2_model_protocol.py` | `8f61e9a75ae6bbb58df7043f4fceaf7b9041fe63f47b722aa4fb77fca745f3f9` |
| `experiments/004_t2_reward_study/t2_seal_v1.json` | `8c04f0c2bc9222f9918829f1e5d922c948583ea03da0eb0174a03e2dfcb314eb` |
| `docs/operations/dual-orchestration/F3_ONE_CALL_PROTOCOL.md` | `893f6c1340924ab38891572ae3407aef7f4731f9059d8999d3e676718de5b178` |

## F3PINR1 verification

| check | result |
|---|---|
| Focused F3 plus F1 predecessor tests | `136 passed in 1.27s` (the existing 117 plus 19 new cases) |
| Adjacent T2 artifact tests | `7 passed in 3.93s` |
| New refusal negatives | second canonical identity claim; sibling intent path; conflicting identity; pre-existing canonical intent; missing/mismatched intent at ingestion; second valid run; same-run provider replay; missing/changed launcher-scope identity in request/result |
| Ruff lint, three changed Python files | passed |
| Ruff format check, three changed Python files | passed |
| Full suite excluding the named reward-lane receipt test | `1801 passed, 18 skipped, 1 deselected, 44 failed in 268.56s`; the count and node IDs exactly match the 44-entry sandbox-baseline ledger |
| Direct 44-node sandbox-ledger replay | all `44` expected nodes failed in `1.18s`; with the full-suite failure cardinality, the sets are equal |
| `git diff --check` | passed |
| Builder wall time | `51m` from lease acquisition through release-byte verification and handoff audit |

## Claim ceiling

This slice establishes Fable-local expected configuration, deterministic packet
content, sealed preconditions, canonical-identity claim/refusal, and ingestion
replay refusal. It is not a reward result. It establishes no candidate service
call, model authentication, candidate admission, simulator behavior, policy
training, protected evaluation, measured feedback, reward improvement, or
humanoid competence. Repository state cannot prove the absence of an unrecorded
external call.
