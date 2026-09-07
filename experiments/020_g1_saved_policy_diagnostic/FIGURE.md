# All paired outcomes: return and survival

- **Input:** the exact independently reproduced Study 020 diagnosis.
- **Output:** paired raw return and observed duration for every declared seed.
- Blue circle: initial checkpoint. Orange square: trained checkpoint. Black X:
  fall. Gray lines join the same noise seed; row offsets only aid legibility.
- Frozen: task, reset, oracle, reward, controller and evaluator. Each policy
  uses its own standard deviations. One training seed, not 16 training replicates.
- Feedback use: compare failure modes before the next separately declared
  oracle or reward proposal; the figure does not select or adopt a policy.
- No smoothing, resampling, normalization, exclusions or uncertainty estimate.
  Early falls shorten the return window. Reaching 20 s is not task success.

## Caption and alt text

All 16 paired action-noise sequences are shown. Falls increase from 5 to 11;
six pairs change from survival to falling and none change in the reverse
direction. Both checkpoints have zero full-task passes. The paired return
decline co-occurs with shorter observation; it is not isolated reward-quality
evidence. These are conditional outcomes at one reset and one training seed.

## Selected local outputs

Directory: `artifacts/gmt/course_configs/study020_figures_20260907/`.

| File | SHA-256 |
|---|---|
| `paired_outcomes_final.png` | `831f5f73550ea08b1c08742eab2a2f92e008cb80bec0860fffdedb9b878f3c34` |
| `paired_outcomes_final.pdf` | `c588b56af0d9ca62746c3de4a8aa1223a4970c43fc5b9272afb2e70f89e9382a` |
| `paired_outcomes_final.csv` | `9c6221a153537445b0cc5dd7a89a22563ed7615805708d59fea7807f68e12d36` |
| `paired_outcomes_final.export.json` | `ee2ab56090f3fb72591378b51f454062d32b9fe9c71d970595de49d32ec3eb8c` |

- PNG inspected at 1760 × 1312; labels, axes, paired markers and caveats visible.
  Different markers supplement color. No accessibility certification claimed.
- Export metadata binds source score, plotting/helper code, settings and
  software versions. Earlier drafts remain local, not selected.
- Isolated Matplotlib 3.11.1 and Pillow 12.3.0; training dependencies unchanged.
- Reproduce with `plot_results.py --score SCORE --output FRESH_BASE
  --export-helper PATH_TO_FIGURE_EXPORT` in a plotting environment.
- Reused the scientific-visualization skill's `figure_export.py`; procedural
  attribution is in [RESULTS.md](RESULTS.md), not a scientific outcome claim.
