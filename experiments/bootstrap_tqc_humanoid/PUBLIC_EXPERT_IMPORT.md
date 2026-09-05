# Public expert actor import

| status | current truth |
|---|---|
| progress | `TASK-20260904-03A2FIX` implements all accepted import-review repairs. The strict actor NPZ and actor-state fingerprint are unchanged; regenerated schema-v2 import and equivalence receipts bind the repaired source and runtime identities. |
| bottleneck | Focused tests pass, but the full suite has one non-baseline failure: the separately committed reward receipt binds the repository-wide source-tree hash. Refreshing that forbidden reward artifact is outside this builder's scope. This remains an `external_base_import` interface check only. |
| next step | Fable reviews this scope-only diff, refreshes and reviews the reward runtime receipt under the reward slice's authority, reruns the full suite, and commits. Only then does slice 03A3 construct the E1 contenders on these exact actor bytes. |

## Artifact identities

| artifact | SHA-256 | bytes | Git status |
|---|---|---:|---|
| metadata registration `research/source_controllers/farama_minari_humanoid_v5_tqc_expert/RECEIPT.json` | `5ba0845e8b0cd9b6f39c956ddc46d0f46e8e08690d8bdf832941a1f914a8e0cf` | 9,395 | metadata only; eligible |
| external import receipt `artifacts/bootstrap_tqc_humanoid/external_actor_import_v1.json` | `c2379a21079c40928548b593cb86a8fd8eea468787cf31c28481792d3121e80b` | 12,465 | local and ignored |
| strict actor NPZ `artifacts/bootstrap_tqc_humanoid/farama_minari_humanoid_v5_tqc_actor_v1.npz` | `60987a4e054db2e04f9cb3ab73e13dfe8e2f3ec7dec46346d2b9d0277ad18d9b` | 618,674 | external payload; local and ignored |
| external equivalence receipt `artifacts/bootstrap_tqc_humanoid/external_actor_equivalence_v1.json` | `0d5ec467051e2eebc16a0a186afbe85cc1364afbeab8fcbefa41f088cb656f31` | 2,687 | local and ignored |

- Source `policy.pth`: SHA-256
  `1e64e56288155087089214548b6a634f332a41955a0d22629efd5ff2240e495c`,
  3,321,462 bytes.
- Actor-state fingerprint:
  `3fd39cc715a10126fd92b20f6ce213c380eb4d5df843a42315aac50cf116748a`.
- Strict NPZ schema SHA-256:
  `f72c9e52ac3771bf635e27edd8f08f69b82313d7dc33700162e42740228ce5c6`.
- The import receipt moved to schema v2 to bind the exact registration receipt,
  importer and worker source identities, reviewed Torch distribution identity,
  bounded worker protocol, and no-overlap tensor-storage contract. The
  equivalence receipt moved to schema v2 to bind verifier and runtime identities
  while excluding ambient RNG-state values from canonical bytes.
- Worker safe-globals observation: 75 entries; sorted qualified-name SHA-256
  `7b70391289d8e8d285612f5ee4db68739af844eae8d945964d1174c83ce7d4b9`;
  module allowlist passed and the ordered qualified-name tuple did not change
  during loading.
- Parked WIP: branch `wip/tqc-v2-attempt-supervisor`, commit `5bdae45`.

## Remaining admission chain

`external_base_import` (packet 03) -> construct both E1 contenders (zero-residual and expanded-TQC initializers) on the imported actor -> locally certified Tier-D references and E2 replay -> E3 identifiability corpus -> E4 tracker-family screen -> freeze the candidate -> E5 causal-use gate. No tracker-admission claim before E5.

## Claim ceiling

The evidence class is `external_base_import`; the evidence level is
`interface_check`. The slice establishes integrity-verified external actor
bytes, a strict code-free NPZ, exact source-to-NPZ output equivalence on one
fixed batch, and payload-free provenance. It is not E1 and establishes no
locomotion, stability, tracking, causal reference use, oracle quality, or other
Humanoid behavior.
