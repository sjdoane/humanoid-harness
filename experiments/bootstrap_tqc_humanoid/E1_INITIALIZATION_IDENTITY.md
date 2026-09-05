# E1 TQC initialization identity

| status | current truth |
|---|---|
| progress | Actual E1 now passes for both ADR 0004 contenders on the integrity-verified imported expert NPZ. Receipt `e1_initialization_identity_external_v1.json` binds the external actor, import/equivalence chain, committed fixture design, pinned observation/reference sets, contender parameters, and paired output hashes. |
| bottleneck | E1 is initialization identity only. Its authority is `external_pretrained_artifact`; it supplies no local-training authority and establishes no behavior, tracker admission, E2, E3, causal reference use, or oracle quality. |
| next step | Fable reviews the exact receipt chain and negative tests. Subsequent tracker work may consume these fixed bytes but must satisfy E2-E5 independently. |

## Two distinct results

| result | input | maximum claim |
|---|---|---|
| Synthetic transfer fixture | Seeded untrained TQC actor; deterministic `4 x 348` observations; nonzero `4 x (8 x 45)` reference input | The expansion and verifier preserve this fixture's action distribution within `atol=rtol=1e-6` |
| Actual E1 gate | Integrity-verified expert NPZ `60987a4e...18d9b`; both declared contender initializers; the pinned fixture observation/reference sets | Bitwise initialization identity for those exact external actor bytes, inputs, and initializers |

The synthetic fixture remains software evidence. The actual E1 result is a
separate receipt constructed from the imported expert, not an upgrade of the
synthetic actor.

## Actual E1 receipt

| binding | recorded identity |
|---|---|
| receipt | SHA-256 `28725ecfba2ca89f4b4608e5e8d5b5024cfe2ed8df71383bc03a248891edf266`; 11,023 bytes |
| expert strict NPZ | SHA-256 `60987a4e054db2e04f9cb3ab73e13dfe8e2f3ec7dec46346d2b9d0277ad18d9b`; actor state `3fd39cc715a10126fd92b20f6ce213c380eb4d5df843a42315aac50cf116748a` |
| expert import/equivalence | `b790f06ccb66ca45809eaa1b0c5cd6804e072c6fd9eb48c61838ee2e558bd9d7` / `65b2090b783e381e799e90e372308018d4e9a50fa6be2df7934af5f24e78a23e` |
| fixture design | SHA-256 `fe0565f2986b70732d1e054fda22c013b80f439a87f8e95cdded6977d438e3e6`; 3,407 bytes |
| observation set | SHA-256 `0c6a81b06a88cab7eca0255e75f021008b60025c4ddc4d3719426e3647159ec6`; `<f4[4,348]` |
| reference set | SHA-256 `ffce030803e735c4d763658df320fec57810fd7c42811797152977538e012d1a`; `<f4[4,360]`, interpreted as `8 x 45` per observation |

### Zero-residual contender

- The deterministic residual mean head contains exact positive-zero weights
  and bias; residual scale is exactly `0.08`.
- Base deterministic action plus `0.08 * 0` is bitwise identical to the base
  deterministic action after normalized clipping and physical-control scaling.
- Normalized control hashes to
  `b19f9bdbbeffd88b528cc574cf82cc5d71fe13a36361e9f5106375d7a138221e`;
  physical control hashes to
  `21db07eb6791ddb669dbb22bc6adc3b76ed6a14cbdb0473d3199135f34040a94`
  on both sides.
- A zero deterministic mean is not zero stochastic exploration. With
  `log_std=-3` and seed `93004`, the recorded residual sample is nonzero and
  hashes to
  `1008c242b2c0b4625a32e16e8fd7be74da398f9cc8c6b2df695f008709cd47ee`.

### Expanded-TQC contender

- The first actor layer expands from `[256,348]` to `[256,708]` for an
  `8 x 45` reference window. All 92,160 appended values are exact positive
  zero.
- Every copied parameter is audited bitwise equal.
- Mean, log standard deviation, deterministic action, and seed-`93003` sample
  are bitwise identical. Their paired hashes are respectively
  `502bebf62b38d3b85cbc653d23460a718edb48a9963f731108da8b3177eab0b8`,
  `a6ae6c3f96d413e2d08b78a31c8f6880a32bc9ff17c0f4d25e3a44c16c019543`,
  `b19f9bdbbeffd88b528cc574cf82cc5d71fe13a36361e9f5106375d7a138221e`,
  and `423f8fd6d4bb1e3ea116e457e3ecf78b197544f03977f9b71de1b3590242cc49`.

Negative tests reject any nonzero residual-mean parameter, nonzero appended
reference column, changed copied parameter, missing observation-set hash, or
missing NPZ hash. Local-training admission APIs reject the E1 receipt.

## Frozen fixture mechanism

```text
hash-pinned x[4,348] ---> exact imported expert actor ---> mean, log_std
          |                                             | deterministic/sample
          |                                             v
          |                                      normalized -> physical
          |
          +-- R[4,8x45] ----------> expanded first layer
                                     [copied state columns | exact-zero R columns]
                                                     |
                                                     +--> bitwise-same outputs

base deterministic action + 0.08 * exact-zero residual mean
          |
          +--> bitwise-same normalized and physical control
```

No critic, target critic, replay buffer, optimizer, entropy state, or ambient
RNG state is transferred. Receipt construction takes no environment or
training step and emits no checkpoint.

The machine-readable design remains
[`configs/tqc_initialization_transfer_fixture_v0.study.json`](configs/tqc_initialization_transfer_fixture_v0.study.json).

## Reproduce the actual E1 receipt

```bash
uv run python -m oracle_composition.experiments.external_tqc_initialization_identity \
  --actor-npz artifacts/bootstrap_tqc_humanoid/farama_minari_humanoid_v5_tqc_actor_v1.npz \
  --import-receipt artifacts/bootstrap_tqc_humanoid/external_actor_import_expert_03a3_v1.json \
  --equivalence-receipt artifacts/bootstrap_tqc_humanoid/external_actor_equivalence_expert_03a3_v1.json \
  --design experiments/bootstrap_tqc_humanoid/configs/tqc_initialization_transfer_fixture_v0.study.json \
  --output artifacts/bootstrap_tqc_humanoid/e1_initialization_identity_external_v1.json
```

The output path is exclusive. The receipt and external payload remain local
and ignored.
