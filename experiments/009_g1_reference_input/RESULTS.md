# Reference-input dependence observed in both tested configurations

| progress | Both five-arm runs passed independent validation and exact positive-control byte comparisons. |
|---|---|
| bottleneck | Input dependence does not establish composition quality or successful task learning. |
| next step | Test the after-only O6 reference revision; evaluate fixed observation normalization in a separately controlled trainer study. |

## Behavioral result

| Configuration | Arm | First practical difference, control boundary | Duration | Fall |
|---|---|---:|---:|---|
| O2 | exact | none | 20 s | no |
| O2 | zero reference | 1 | 1.6 s | yes |
| O2 | current frame repeated | 1 | 20 s | no |
| O2 | shuffled reference | 1 | 5.1 s | yes |
| O2 | shifted reference | 1 | 20 s | no |
| O4b four-state | exact | none | 20 s | no |
| O4b four-state | zero reference | 1 | 1.6 s | yes |
| O4b four-state | current frame repeated | 1 | 20 s | no |
| O4b four-state | shuffled reference | 1 | 20 s | no |
| O4b four-state | shifted reference | 1 | 20 s | no |

- All four treatments change the first action (command index 0). Practical
  trajectory differences exceed the locked 1 mm / 1 mrad thresholds at the
  first subsequent boundary, well within the predicted first 50 steps.
- Original oracle metadata first differs at command 94 for the surviving O2
  treatments and command 92 for O4b. Zero-reference arms fall before an oracle
  metadata difference is observed. The intervention itself starts at command 0.
- Both exact arms reproduce all three retained positive-control files:
  trajectory, frame ledger and evaluation. No metadata exceptions were needed.
- All five arms within each run have identical initial `qpos` and `qvel` and
  the expected fixed-home reset. Actual actor inputs, history, original oracle
  commands, interventions, actor outputs, zero residuals and post-step targets
  pass full data-only reconstruction after the supervisor publishes its receipt.

## Interpretation

- This closes the **reference-window input-dependence gate for these two
  configurations and interventions**. It is more than matched-state action
  sensitivity: the actual robot trajectories change in closed loop.
- Surviving corrupted-input arms are not task wins. Corruption need not cause
  a fall to demonstrate input dependence.
- Shifted-arm mean joint error is lower against its actor-window first row
  than against the original objective target: O2 `0.0999 vs 0.1820 rad`, O4b
  `0.0966 vs 0.2717 rad`. These are descriptive errors on each arm's evolving
  trajectory, not a separate randomized tracking-quality estimate.
- No learning, held-out generalization, reference-composition quality or full
  task success is established. Comparisons after an arm falls are unavailable.
- Sensor and 1 kHz contact/saturation records retain their producer-evidence
  limitation. The validator does not regenerate MuJoCo physics.

## Conditioning diagnostic, not a safety threshold

| Exact-arm base observation | O2 | O4b |
|---|---:|---:|
| Raw maximum absolute value / RMS | 3.411 / 0.424 | 4.712 / 0.478 |
| Pinned actor-normalized maximum / RMS | 9.044 / 0.859 | 9.035 / 0.954 |

- The exact actor transform uses its retained mean/std and epsilon `1e-4`.
  No observed exact-arm value exceeds magnitude 10 after normalization.
- These are frozen-base visited states, not trained-residual distributions.
  Neither these ranges nor arbitrary magnitude cutoffs establish safety or
  that normalization improves learning. An explicit matched study is required.

## Receipts

- Source: `bce2847c5cb51bfdc14cd63beabd405ca2a7fd0a`; 192 files;
  tree `17c52ac2b238bbe4dfecbe23f244c0fe027a6592ea77e251057763b0ea2743f8`.
- Parent focused suite: 510 passed; independent Fable suite: 516 passed.
  Different selections; neither is a claim that the whole repository passes.
- Runs are under sibling `humanoid-harness-probe-runs/`.

| Artifact | SHA-256 |
|---|---|
| `gmt_reference_ablation_o2_20260907/reference_ablation_manifest.json` | `711875494abc39d4751d2e46be552fef8ec8efcdd99307bf77b0d0a87632d144` |
| O2 resource receipt | `405858d237a22c316d422b8d5d240d8b0d2d44be70460ffa661240cdb2da7a2b` |
| `gmt_reference_ablation_o4b_20260907/reference_ablation_manifest.json` | `3e6398e7696b5441c27b7de616a429908389706da0208001ddcd2a68c7dffa10` |
| O4b resource receipt | `44b9e11d072acf682ee0a72bad00c0af7785b419416c6da3aae16ebe220dd757` |
| Local `study009_o2_verified_readout_20260907.json` | `21dd3073188bce2f959011a058b3e44cc7f8045b33847bbae38220f0f55ccd36` |
| Local `study009_o4b_verified_readout_20260907.json` | `4cd791b37f312e001e5cb18dc419069f4fbd88b98935fb10c73a7378496cfdef` |

The readouts are under `artifacts/gmt/course_configs/`; each includes full
per-arm objectives, both reference-error definitions and common-horizon
divergence. Both resource slots were released after successful validation.
