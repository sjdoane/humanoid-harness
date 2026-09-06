# GMT G1 admission slice

This package converts one exact upstream GMT deployment checkpoint and eight
exact motion files without interpreting their pickle programs. It reads only
declared raw tensor spans after SHA-256, size, ZIP-member, graph-member, dtype,
shape, stride, and finiteness checks.

```bash
PYTHONPATH=src python -m oracle_composition.adapters.gmt bundle \
  /path/to/pinned/gmt/repo /path/to/new/output
```

The weight artifact is a deterministic numeric-only NPZ. Loading requires its
post-conversion SHA-256:

```python
from oracle_composition.adapters.gmt.actor import load_actor

actor = load_actor(path, expected_sha256=receipt_sha256, freeze=True)
```

## Frozen deployment ABI

- Input: `2154 = 20x30 reference + 74 current proprioception + 20x74 history`.
- Output: 23 raw joint offsets at 50 Hz; preserve the raw output in history
  before clipping to `[-10, 10]`, scaling by `0.5`, and adding the default pose.
- Actor: two temporal encoders plus a `296-1024-1024-512-256-23` SiLU MLP;
  LayerNorm follows the 256D linear layer.
- Normalization: `(observation - mean) / (std + 1e-4)` in float32.
- Motions preserve native cadence and non-unit quaternions exactly; the
  converter does not normalize, retime, smooth, or derive velocities.

## Claim and rights boundary

Conversion and plain-actor inference are implemented. Original-JIT numerical
equivalence, MuJoCo replay, Mac throughput, tracking competence, training, and
fine-tuning stability remain unmeasured. The upstream repository is pinned at
`2a590de25a1eb08e47491977a738549c22f16e1f`; its root is Apache-2.0, while the
README says research use only and supplies no separate weight or per-motion
license. Converted bytes remain local and must not be redistributed until those
terms and motion provenance are resolved.
