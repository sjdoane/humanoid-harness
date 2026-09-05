# Public TQC actor imports

| status | current truth |
|---|---|
| progress | `TASK-20260904-03A3` integrity-verified and imported the public `expert`, `medium`, and `simple` actors through the same closed, data-only importer. Every actor has a registration, schema-v2 import receipt, strict code-free NPZ, and fixed-batch source-equivalence receipt. Actual E1 initialization identity also passes on the imported expert bytes for both ADR 0004 contenders. |
| bottleneck | These are external artifacts and interface checks. No training or behavior evaluation ran, and no tracker, E2, E3, causal-reference-use, or oracle claim is available. |
| next step | Fable reviews the registrations, importer profiles, negative tests, receipt chain, and full-suite baseline comparison before committing. Later tracker work must retain the exact imported-expert and E1 identities and proceed through E2-E5. |

## Source registrations

All 18 present files in the `medium` and `simple` local artifact copies were
verified before any load. Large-file identities use the API record's
`lfs.sha256`; small-file identities use Git blob SHA-1 from `blobId`. The
registration receipts keep both algorithm labels explicit.

| variant | repository commit | `policy.pth` SHA-256 | bytes | source-reported training facts | registration receipt SHA-256 |
|---|---|---|---:|---|---|
| `expert` | `e5a86ffdb70e6f4750f39c0464ac026a8437001a` | `1e64e56288155087089214548b6a634f332a41955a0d22629efd5ff2240e495c` | 3,321,462 | existing registration | `5ba0845e8b0cd9b6f39c956ddc46d0f46e8e08690d8bdf832941a1f914a8e0cf` |
| `medium` | `949f7963c1a8964587dca73d48873ad021e168b5` | `d54c93dd82d97cd931caf85fcba5b4722b9c86075e9c679576b6ba45b7823c66` | 3,321,462 | 4,950,000 of 5,000,000 timesteps; `8021.95 +/- 912.19` mean reward | `68344e980e42543ddd5ca3d164a9c8173f63950d47b4c8c9064e57283fd4e0a1` |
| `simple` | `39e2954c193fc1352535935a7d71eca9e5745d9b` | `ca9aff0dc359d6011ddde33f196fc8b612781cba80247fb8a9889eb1df74e58e` | 3,321,014 | 1,965,000 of 2,000,000 timesteps; `5543.17 +/- 825.31` mean reward | `745d82dea8fc4878f70ed58f4d6ff4cd506b0f94c85676dc70e3cffce8283ac5` |

For both new siblings, the source records report Stable-Baselines3 `2.4.1`,
seed `0`, and `policy_kwargs {"use_sde": false}`. The reward values remain
source-reported metadata with `verified: false`; this slice did not reproduce
them. The license remains unspecified, technical import approval is not a
redistribution grant, use is limited to local development, and no payload bytes
may enter Git.

## Import and equivalence receipts

| variant | import receipt SHA-256 | strict NPZ SHA-256 (bytes) | actor-state SHA-256 | equivalence receipt SHA-256 |
|---|---|---|---|---|
| `expert` | `b790f06ccb66ca45809eaa1b0c5cd6804e072c6fd9eb48c61838ee2e558bd9d7` | `60987a4e054db2e04f9cb3ab73e13dfe8e2f3ec7dec46346d2b9d0277ad18d9b` (618,674) | `3fd39cc715a10126fd92b20f6ce213c380eb4d5df843a42315aac50cf116748a` | `65b2090b783e381e799e90e372308018d4e9a50fa6be2df7934af5f24e78a23e` |
| `medium` | `b1c07c2f5070b48ff6bb28983b16ed16d1616e7ad18f557639dad89bb1f4f172` | `2677ebb70cd20e0ba8f8a591814fc853e325277db65184ebafb5bf5e21198b04` (617,761) | `cffb5679e3ef10cc99980819013941cb55cdc9f2198ba4a91a306dfc5f73e00c` | `ede73be8db0ec3411dfb1a5ce2dbbe109d49432d1aeba97e6e4b9bf4fb44d487` |
| `simple` | `bef2a2678d30f13a9e3350bf8c8f591661bb129b063276a15b8051f94ee313f7` | `b09aa921316640024e9703671d328d7ecbabe76f04cfed8c1d8fb86fbd95a917` (618,054) | `3068049a0d18b7924d117a96aafee7127e411a70eba344ff985d31e97da3312b` | `604db637d316d3b8e46fb6f5d7a9b9004cfced267836394d89df183c87615268` |

Every equivalence receipt passes exact mean, log-standard-deviation,
deterministic-action, and shared-seed sampled-action equality on the pinned
`4 x 348` batch. The strict NPZ schema is unchanged and hashes to
`f72c9e52ac3771bf635e27edd8f08f69b82313d7dc33700162e42740228ce5c6`.
The importer rejects a mismatched hash, byte count, or key schema before
export.

## Actual E1 row

| artifact | result | receipt SHA-256 | bytes |
|---|---|---|---:|
| `e1_initialization_identity_external_v1.json` | both contenders pass bitwise initialization identity on expert NPZ `60987a4e...18d9b` | `28725ecfba2ca89f4b4608e5e8d5b5024cfe2ed8df71383bc03a248891edf266` | 11,023 |

The E1 receipt binds the NPZ, fixture design, pinned observation and reference
sets, expert import and equivalence receipts, contender parameters, and every
paired output hash. It is labeled `external_pretrained_artifact`, has zero
training and environment steps, and is ineligible for local-training or
tracker-admission authority.

## Admission chain and claim ceiling

`external_base_import` -> E2 replay -> E3 identifiability corpus -> E4
tracker-family screen -> freeze the candidate -> E5 causal-use gate. No
tracker-admission claim exists before E5.

The evidence class remains `external_base_import`. This slice establishes
integrity-verified external actor bytes for three public policies and actual E1
initialization identity on the imported expert. It establishes no Humanoid
behavior, tracker quality, reference use, E2, E3, or oracle quality.
