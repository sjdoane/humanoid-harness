# DeepMimic humanoid3d source motions

| | Status |
|---|---|
| **research target** | Candidate source motion for later controller-native retargeting and phase/oracle tests |
| **implemented capability** | Exact upstream bytes plus a bounded, deterministic source-format audit bound to their manifest and SHA-256 identities |
| **measured evidence** | Walk, run, and face-up get-up pass the frozen source-format checks; face-down get-up fails them; none has Gymnasium mapping, tracking, kinematic admission, or a dynamics certificate |

These four data-only files come from the official
[DeepMimic repository](https://github.com/xbpeng/DeepMimic). The bytes match
upstream commit `1f915c52fcd4b95b5f5f15b759ae91bd81e9a801` and the copy retained in the
read-only legacy repository at commit
`c1779779f20f418a5ddf78576fdb689895f1e108`. The accompanying MIT license is
preserved verbatim.

## Admission boundary

DeepMimic `humanoid3d` and Gymnasium `Humanoid-v5` do not share an asserted
joint, root-frame, cadence, or embodiment contract. These files are therefore
source candidates only. They must not enter tracker training until a new
content-addressed artifact:

1. defines the complete coordinate and joint mapping;
2. preserves root-frame and cadence semantics;
3. passes exact structural and kinematic checks; and
4. earns a separate same-simulator dynamics-feasibility certificate.

Copying or parsing a source file does not satisfy any of those gates. The
static positive-control experiment does not consume these bytes.

## Source-format audit

Regenerate [`source_format_audit.json`](source_format_audit.json) with:

```bash
uv run python scripts/inspect_deepmimic_sources.py --write
```

Use the same command with `--check` to verify the committed receipt without
changing it.

The audit checks bounded strict JSON, exact 44-value frames, durations, and all
nine documented quaternion fields. At the frozen 2% quaternion-norm tolerance,
the face-down get-up source fails at the neck (maximum error 2.340861%) and
right ankle (2.119944%). It also has a left-shoulder quaternion sign
discontinuity between source frames 4 and 5. The audit normalizes temporary
quaternion copies only to calculate adjacent-frame dot products; it neither
changes the retained bytes nor emits an admitted motion.
