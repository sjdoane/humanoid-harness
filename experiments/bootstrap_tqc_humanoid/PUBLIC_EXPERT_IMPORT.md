# Public expert actor import

| status | current truth |
|---|---|
| progress | The pinned Farama `Humanoid-v5` TQC `policy.pth` was imported through one isolated weights-only load into the strict actor NPZ, and source-to-NPZ outputs matched exactly on the fixed `4 x 348` batch. |
| bottleneck | This is an `external_base_import` interface check only. Neither E1 contender has been constructed; no tracker, reference-use result, or Humanoid behavior is admitted. |
| next step | Slice 03A3 constructs both E1 contenders on these exact actor bytes. Slice 03B performs the separate 20-reset development screen and replay-bundle work; that screen has not run. |

## Artifact identities

| artifact | SHA-256 | bytes | Git status |
|---|---|---:|---|
| metadata registration `research/source_controllers/farama_minari_humanoid_v5_tqc_expert/RECEIPT.json` | `5ba0845e8b0cd9b6f39c956ddc46d0f46e8e08690d8bdf832941a1f914a8e0cf` | 9,395 | metadata only; eligible |
| external import receipt `artifacts/bootstrap_tqc_humanoid/external_actor_import_v1.json` | `ce90c312f7c222847a936edd2d964d386bab01d1acb0fd924de3a8db951430cb` | 11,308 | local and ignored |
| strict actor NPZ `artifacts/bootstrap_tqc_humanoid/farama_minari_humanoid_v5_tqc_actor_v1.npz` | `60987a4e054db2e04f9cb3ab73e13dfe8e2f3ec7dec46346d2b9d0277ad18d9b` | 618,674 | external payload; local and ignored |
| external equivalence receipt `artifacts/bootstrap_tqc_humanoid/external_actor_equivalence_v1.json` | `d57eedf35ded4dc5e1f9e6f1cb04c9a67f5db1b491d1825ae6a2196099fc6d32` | 2,429 | local and ignored |

- Source `policy.pth`: SHA-256
  `1e64e56288155087089214548b6a634f332a41955a0d22629efd5ff2240e495c`,
  3,321,462 bytes.
- Actor-state fingerprint:
  `3fd39cc715a10126fd92b20f6ce213c380eb4d5df843a42315aac50cf116748a`.
- Strict NPZ schema SHA-256:
  `f72c9e52ac3771bf635e27edd8f08f69b82313d7dc33700162e42740228ce5c6`.
- Worker safe-globals observation: 75 entries; sorted qualified-name SHA-256
  `7b70391289d8e8d285612f5ee4db68739af844eae8d945964d1174c83ce7d4b9`;
  module allowlist passed and the list did not change during loading.
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
