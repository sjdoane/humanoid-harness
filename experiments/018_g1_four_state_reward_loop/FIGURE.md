# Recorded height versus reference target

- **Question:** when the robot occupies the physical crouching region, is its
  reference target inside the posture band, and does the robot follow it?
- Two aligned panels: zero residual and trained A. Solid black is actual root
  height; dashed blue is the retained post-step target. Hatching marks measured
  progress in[1,2), not oracle mode. Dotted orange marks the0.60 m ceiling;
  dash-dot gray marks the separate minimum-depth target.
- Display: first8 s of each20 s episode, identical0.30–1.05 m axes. All displayed
  heights fit the axes, including the initial settling transient. No smoothing,
  interpolation, normalization or uncertainty estimate; one episode per policy.
- CSV retains all2000 samples across both complete episodes. Counts and minimum
  region heights reproduce the [result table](RESULTS.md).
- Interpretation: timing/reference conflict and imperfect tracking coexist.
  The figure does not isolate a cause or establish dynamic unreachability.

## Local outputs

Sibling directory: `humanoid-harness-probe-runs/`.

| Output | SHA-256 |
|---|---|
| `study018_height_diagnostic_final_20260907.png` | `5d5db03f1ae9c54e3731acc4e99a221eb1ec4e4a417eb5eb4bec382589d0f967` |
| `study018_height_diagnostic_final_20260907.pdf` | `4c7dac9422a9bff239f2d4f5bd1e750eed0df13e5d4eafd39b91fc8d3fa0097f` |
| `study018_height_diagnostic_final_20260907.csv` | `71e91d88ee74fc611eaeaa2cfa8c833a0484606eef654c8395ebe0f18274b354` |

- `.export.json` binds raw-input hashes, plotting/helper code hashes, versions,
  transformations, alt text and display scope. Raw inputs are unchanged.
- PNG inspected at1600×1056 pixels. Alpha channel is uniformly255: opaque.
  Styles/hatching supplement color. Manual inspection is not accessibility or
  publisher certification. Earlier render drafts remain local, not selected.
- Reproduce with `plot_diagnosis.py --run RUN --output FRESH_BASE
  --export-helper PATH_TO_FIGURE_EXPORT`. Rendering used isolated
  Matplotlib3.11.1 and NumPy2.2.6; project simulation dependencies were unchanged.

## Software guidance

Scientific-visualization skill informed provenance, shared axes, redundant
encodings and final-size inspection. Its supplied `figure_export.py` was reused.
Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner, A. M. (2026).
[Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065).
Current record v2 checked2026-09-07. This is tooling attribution, not evidence
that the humanoid performs better.
