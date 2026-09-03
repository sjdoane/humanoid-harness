# Experiment 002A: matched-observation reference-input probe

| status | update |
|---|---|
| progress | Four deterministic input transforms and actor/critic consumption receipts are implemented. |
| bottleneck | The current reference-conditioned checkpoint falls and is not admitted. |
| next step | Run this local probe, then train a stable tracker before behavioral Experiment 002B. |

## Question

- Does the exact local checkpoint change its output when only its numeric
  `8 x 45` reference window changes?
- Claim ceiling: numeric input sensitivity for this checkpoint.
- This is not evidence of stable tracking, oracle quality, or useful behavior.

## Frozen mechanism

```text
pinned Minari episode-0 observation x[t] -+--------------------+
                                          |                    |
pinned 45D reference R[t:t+8] -> transform arm                 |
                     exact / zero / shuffle / shift            |
                                          |                    |
                                          v                    v
                              frozen actor + critic       frozen base actor
                                      |     |                    |
                             action bytes  value          base-action bytes
                                      \     |                   /
                                       float32 composition
                                                |
                                       composed-control bytes

Ground-truth R[t:t+8] stays unchanged in every arm.
No simulator state is restored, no action is submitted, and no generated reward
enters this probe.
```

## Predeclared local design

| item | value |
|---|---|
| Source observation/reference | Registered Minari `Humanoid-v5` expert episode 0; Tier K; local only |
| Snapshot frames | `0, 125, 250, 375, 500, 625, 750, 875` |
| `C_exact` | Exact terminal-hold window |
| `C_zero_input` | Positive `float64` zero bytes |
| `C_shuffle_input` | Full-sequence SHA-256 ranking; seed `260907`; then window |
| `C_shift_input` | Full-sequence cyclic source shift `+250`; then window |
| Actor/critic proof | First-layer input hooks must match the declared normalized `float32` bytes |
| Repeatability | Duplicate inference must match bit-for-bit |
| Gate | Every corrupted arm changes input, actor output, and composed control at all eight observations |

## Matched fields

- Source observation bytes.
- Unmodified ground-truth window bytes.
- Base-action bytes.
- Checkpoint and normalizer state.
- Feed-forward recurrent state: empty.
- Residual scale and float32 composition rule.

Only the controller-visible reference window may change.

## Interpretation

| result | allowed statement |
|---|---|
| Gate fails | This checkpoint has not shown reference-input sensitivity. |
| Gate passes | This checkpoint's composed control depends on the numeric reference window at the eight matched source observations. |
| Experiment 002B passes later | Correct reference input improves protected tracking behavior for the admitted tracker family. |

The eight observations are repeated measurements on one existing checkpoint. They
are not eight independent training replicates. No p-value or population claim
is reported.

## Run

```bash
humanoid-harness tracker probe-reference-use \
  --output artifacts/experiment_002a/reference_causal_probe.json
```

The command reads only registered, hash-pinned local data and strict data-only
NPZ controllers. It refuses to overwrite an existing report.

## Remaining gate

- The current checkpoint is one local, incompletely documented seed.
- Its reference is Tier K and has unresolved redistribution rights.
- Closed-loop exact-reference behavior collapses.
- No complete simulator state is stored or restored, so this does not satisfy
  Experiment 002B's formal matched-state or one-step physical intervention.
- Experiment 002B therefore requires a newly trained, stable, admitted tracker
  and a separately locked expected-direction behavioral analysis.
