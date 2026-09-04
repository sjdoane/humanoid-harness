# E1 TQC initialization identity

| | |
|---|---|
| progress | A hash-pinned synthetic actor-expansion fixture and reusable numeric verifier are implemented. |
| bottleneck | E0 discarded its model by design; no trained TQC actor bytes exist for the actual E1 gate. |
| next step | Train and retain one trusted local TQC actor under a separate reviewed protocol, then verify these same outputs on its pinned bytes. |

## Two distinct results

| result | input | maximum claim |
|---|---|---|
| Synthetic transfer fixture | Seeded untrained TQC actor; deterministic `4 x 348` observations; nonzero `4 x (8 x 45)` reference input | The expansion and verifier preserve this fixture's action distribution within `atol=rtol=1e-6` |
| Actual E1 gate | One trusted, trained, SHA-256-pinned TQC actor and one separately pinned observation set | Initialization identity for those exact controller bytes |

Passing the fixture leaves the actual E1 gate false. It does not establish
balance, locomotion, tracking, reference use, or oracle quality.

Actual E1 requires both paths to run on the same persisted trained actor:

- frozen deterministic TQC plus the declared zero-mean residual initializer;
- the expanded-TQC initializer derived from those exact actor bytes.

Neither actual initialization has run.

## Fixture mechanism

```text
hash-pinned x[4,348] ---> seeded synthetic TQC actor ---> mean, log_std
          |                                             | deterministic/sample
          |                                             v
          |                                      normalized -> physical
          |
          +-- nonzero R[4,8x45] --> expanded first layer
                                     [copied state columns | exact-zero R columns]
                                                     |
                                                     +--> same recorded outputs

base deterministic action + 0.08 * exact-zero residual mean
          |
          +--> bitwise-identical normalized and physical control

The synthetic residual has log_std=-3. Its seeded stochastic sample is
recorded and nonzero; a zero deterministic mean is not zero exploration.
```

## Frozen checks

- Bind fixture ID `tqc_actor_expansion_transfer_fixture/v1` to exactly one
  design artifact:
  - SHA-256 `fe0565f2986b70732d1e054fda22c013b80f439a87f8e95cdded6977d438e3e6`;
  - byte count `3,407`;
  - the loader rejects re-encoding, missing fields, alternate values, and
    non-JSON numeric types.
- Require the exact successful E0 receipt:
  - receipt SHA-256 `2436c2e93cac8b5ed357acdcb19cca0b6222792a1c9c0d65f06577509d68a587`;
  - runtime SHA-256 `710d524b2936ee29bbeb5be9336ed5bb0e12964919f7035d213d9f19348e0a77`;
  - `100,000` environment steps;
  - `20,000` vector calls;
  - `19,980` updates;
  - authoritative resource gate pass;
  - no checkpoint, replay, normalizer, or controller artifact emitted.
- Pin the observation and reference-set hashes in the design.
- Pin NumPy, PyTorch, Stable-Baselines3, SB3-Contrib, dependency-lock, actor,
  distribution, base-policy, torch-layer, adapter, contract, and runner source
  identities.
- Record Gaussian mean and `log_std`, deterministic normalized action, shared-
  seed sampled normalized action, and normalized-to-physical control.
- Record the residual, unclipped sum, clipped action, saturation count, and
  lost residual authority separately.
- Recompute the fixture inside receipt construction. Reject caller records when
  their sampling noise, action tensors, saturation count, lost-authority
  tensors, actor tensor-value hashes, runtime identity, or source identities
  differ from the canonical execution.
- Require copied actor parameters to be bitwise equal, except for the expanded
  first layer's exact-zero reference columns.
- Transfer no critic, target critic, replay buffer, optimizer, or entropy state.
- Emit no checkpoint and take no environment step.

The machine-readable design is
[`configs/tqc_initialization_transfer_fixture_v0.study.json`](configs/tqc_initialization_transfer_fixture_v0.study.json).

## Run the software fixture

```bash
uv run humanoid-harness tracker verify-tqc-transfer-fixture \
  --design experiments/bootstrap_tqc_humanoid/configs/tqc_initialization_transfer_fixture_v0.study.json \
  --e0-receipt artifacts/bootstrap_tqc_humanoid/resource_calibration_seed_92001.json \
  --output artifacts/bootstrap_tqc_humanoid/tqc_initialization_transfer_fixture_v0.json
```

The output path is exclusive. The output remains local and ignored.

## Actual E1 remains blocked

- The resource probe cannot supply controller bytes: its controller was
  deliberately discarded without serialization.
- Do not substitute the synthetic actor's parameter hash for a trained
  checkpoint identity.
- Do not load uploaded SB3, pickle, Torch, or raw checkpoint objects.
- A future execution adapter should consume the strict actor-only NPZ defined
  for the one-million-step development screen. It must bind the archive,
  semantic actor-state, schema, source checkpoint, and normalizer identities,
  then rerun the verifier without changing the pinned observation set or
  tolerances.
- The fixture receipt's `fixture_execution_authoritative` field applies only to
  this software fixture. It does not grant actual E1 or behavioral authority.
