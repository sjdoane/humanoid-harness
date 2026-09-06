# GMT matched-state reference sensitivity

| State | Evidence boundary |
|---|---|
| progress | Real matched-state test completed: exact actions reproduce byte-for-byte; zero/shuffle/shift change all 500 action rows. |
| bottleneck | No closed-loop intervention rollout has been run. |
| next step | Keep this as input-influence evidence; use the separate course trials for composition and learning claims. |

- Mean raw-action L2 changes: zero `4.94323`, shuffle `2.80658`, shift `3.01661`.
- Retained non-reference dimensions: 1,554, unchanged per matched state.
- Output in sibling `humanoid-harness-probe-runs/`:
  `gmt_walk_stand_reference_sensitivity_20260906.npz`.
- SHA-256: `7e7499b920e14dec3ca94170b959b292db057fa64b8d928d6d2642db6244c8c6`.

## Question

- Does the reconstructed GMT actor's raw action change when only its 600 reference dimensions change at the same recorded robot state?
- This is an input-influence check. It is not a reference-composition, survival, tracking, learning, or task-success result.

## Fixed and changed inputs

| Condition | Reference dimensions `0:600` | Dimensions `600:2154` |
|---|---|---|
| exact | retained reference window | retained bytes |
| zero | `+0.0` | retained bytes |
| shuffled | one seeded PCG64 permutation of all 20 frames | retained bytes |
| shifted | admitted `motion.window(control_step + 250)` | retained bytes |

- `+250` control steps is `+5.0 s` at the pinned `50 Hz` controller cadence.
- The command fails if replayed exact actions differ byte-for-byte from retained raw actions.
- The command fails if trace, manifest, actor, motion, model, mesh, clock, observation, or history lineage differs.

## Run

```bash
PYTHONPATH="$PWD/src" python -m oracle_composition.adapters.gmt reference-sensitivity \
  --trace /absolute/path/gmt_g1_walk_stand_10s_trace.npz \
  --trace-sha256 TRACE_SHA256 \
  --manifest /absolute/path/gmt_g1_walk_stand_10s_trace.npz.manifest.json \
  --manifest-sha256 MANIFEST_SHA256 \
  --upstream-root /absolute/path/GMT \
  --weights /absolute/path/gmt_g1_actor_weights.npz \
  --weights-sha256 WEIGHTS_SHA256 \
  --motion /absolute/path/walk_stand.npz \
  --motion-sha256 MOTION_SHA256 \
  --motion-name walk_stand \
  --shuffle-seed 20260906 \
  --output /new/path/gmt_walk_stand_reference_sensitivity.npz
```

## Outputs

- Numeric NPZ:
  - retained and recomputed exact raw actions;
  - zeroed, shuffled, and shifted-reference raw actions;
  - shuffle permutation;
  - retained and shifted-source control steps.
- Immutable JSON manifest:
  - exact input hashes and source identities;
  - intervention provenance;
  - descriptive per-state action-delta summaries;
  - explicit claim limits.
